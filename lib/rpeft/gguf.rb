# frozen_string_literal: true

require 'find'
require 'json'
require 'open3'
require 'tmpdir'

module RPEFT
  # Exports a merged model directory to a single GGUF file for llama.cpp, Ollama, and other local runtimes.
  #
  # The conversion is left to llama.cpp's own tools, which know every architecture and tokenizer GGUF supports:
  # its `convert_hf_to_gguf.py` writes 16-bit or Q8_0 files, and `llama-quantize` turns a 16-bit file into
  # the K-quant types. RPEFT checks the input, runs the tools, and cleans up after them.
  module GGUF
    CONVERTER = 'convert_hf_to_gguf.py'
    QUANTIZER = 'llama-quantize'
    METADATA_EDITOR = File.join('gguf-py', 'gguf', 'scripts', 'gguf_new_metadata.py')
    CHAT_TEMPLATE_KEY = 'tokenizer.chat_template'
    # The list of a model's named templates, such as one for tool use. The templates themselves sit under keys
    # holding their names, which only a reader of the file would know, so they stay; the default one is what Ollama was seen to use.
    CHAT_TEMPLATE_NAMES_KEY = 'tokenizer.chat_templates'
    MAGIC = 'GGUF'

    # Types the converter writes directly, by its --outtype name.
    CONVERTED = {f16: 'f16', bf16: 'bf16', q8_0: 'q8_0'}.freeze
    # Types that need a 16-bit file run through llama-quantize, by its type name.
    QUANTIZED = {q6_k: 'Q6_K', q5_k_m: 'Q5_K_M', q4_k_m: 'Q4_K_M'}.freeze
    TYPES = [*CONVERTED.keys, *QUANTIZED.keys].freeze
    MAX_JSON_BYTES = 256 * 1024 * 1024

    # Runs a command given as an argument list, never through a shell, with extra environment variables,
    # and returns its combined output and whether it succeeded. Replaced in tests so that no external tool runs.
    DEFAULT_RUNNER = lambda do |argv, env|
      output, status = Open3.capture2e(env, *argv)

      [output, status.success?]
    end

    class << self
      attr_writer :runner

      def runner
        @runner || DEFAULT_RUNNER
      end

      def export(model_dir, output, quantize: :q8_0, chat_template: true, llama_cpp: nil, llama_quantize: nil)
        type = quantize_type(quantize)
        raise ConfigurationError, "chat_template must be true or false, got #{chat_template.inspect}" unless [true, false].include?(chat_template)

        model_dir = model_directory(model_dir)
        output = output_path(output)
        llama_cpp_dir = llama_cpp_directory(llama_cpp)
        converter = converter_path(llama_cpp_dir)
        quantizer = quantizer_path(llama_quantize, llama_cpp_dir) if QUANTIZED.key?(type)
        editor = metadata_editor_path(llama_cpp_dir) unless chat_template

        # The tools write into a directory of their own next to the output, under fixed names: the converter
        # expands placeholders such as {ftype} in the name it is given, so it never sees the caller's name.
        # Only a verified file reaches the output, and everything else goes with the directory, including the bf16
        # intermediate of the K-quant types, which is as large as the model and so belongs on the output's disk.
        Dir.mktmpdir('.rpeft-gguf-', File.dirname(output)) do |dir|
          staged = File.join(dir, 'model.gguf')
          env = converter_env(dir)

          if quantizer
            intermediate = File.join(dir, 'model-bf16.gguf')
            convert(model_dir, intermediate, 'bf16', converter, env)
            run([quantizer, intermediate, staged, QUANTIZED.fetch(type)], QUANTIZER)
          else
            convert(model_dir, staged, CONVERTED.fetch(type), converter, env)
          end

          verify(staged, quantizer ? QUANTIZER : CONVERTER)
          staged = remove_chat_template(staged, editor, env) if editor

          publish(staged, output)
        end

        output
      rescue SystemCallError => e
        raise ExportError, "GGUF export to #{output} failed: #{e.message}"
      end

      # The llama.cpp tools as RPEFT would find them now, for `rpeft doctor`. Never raises.
      def tools(env: ENV)
        configured = env.fetch('LLAMA_CPP_DIR', '').to_s
        dir = configured.empty? ? nil : expand(configured) || configured
        converter = dir && File.join(dir, CONVERTER)

        {
          'llama_cpp_dir' => dir,
          'converter' => converter && File.file?(converter) ? converter : nil,
          'quantizer' => find_quantizer(dir, env)
        }
      end

      private

      def quantize_type(value)
        type = value.to_s.downcase.to_sym if value.is_a?(String) || value.is_a?(Symbol)
        return type if TYPES.include?(type)

        raise ConfigurationError, "quantize must be one of #{TYPES.join(', ')}, got #{value.inspect}"
      end

      def model_directory(value)
        dir = expanded_path(value, 'model')

        unless File.directory?(dir)
          raise ConfigurationError, "GGUF export needs a local model directory, got #{dir.inspect}; save a Hub model or merge an adapter into a directory first"
        end

        if File.file?(File.join(dir, 'adapter_config.json'))
          raise ConfigurationError, "#{dir} is a LoRA adapter; merge it into its base model before exporting to GGUF"
        end
        raise ConfigurationError, "#{dir} has no config.json, so it is not a Transformers model directory" unless File.file?(File.join(dir, 'config.json'))

        refuse_custom_code(dir)

        dir
      end

      # A model RPEFT merged ships no custom code, so a directory that does is refused rather than trusted:
      # Python files, and auto_map in any JSON file, since a config can name further config files
      # (configuration_files) that Transformers reads instead. Extensions are compared in any case, as a file
      # system such as macOS's default one finds configuration_x.PY when Transformers asks for
      # configuration_x.py. Find covers hidden directories too and does not follow links.
      #
      # The check runs once, and the converter reads the directory again later, so a directory someone else can
      # write to in between is not covered. Checking a private copy instead would cost a full copy of the model,
      # for a directory that is normally a model the user merged; the README asks for such a directory instead.
      def refuse_custom_code(dir)
        # The model directory itself may be reached through a link, which Find would not enter.
        root = File.realpath(dir)

        Find.find(root) do |path|
          next if path == root

          name = path.delete_prefix("#{root}/")
          refuse_entry(dir, root, name, path)
          extension = File.extname(path).downcase
          raise ConfigurationError, "#{dir} contains Python code (#{name}), which the GGUF converter could run" if extension == '.py'

          refuse_auto_map(dir, name, path) if extension == '.json' && File.file?(path)
        end
      end

      # The converter reads whatever a name in the directory leads to and writes some of it, such as the chat
      # template, into the GGUF file. A link leading out of the directory, or a hard link, could thus carry
      # another file, such as a credential, into a GGUF file that is then published, and a special file such
      # as a FIFO could stall the converter. A link to a directory is refused because Find does not enter it.
      def refuse_entry(dir, root, name, path)
        stat = File.lstat(path)
        if stat.symlink?
          refuse_link(dir, root, name, path)
        elsif stat.file?
          raise ConfigurationError, "#{dir} holds #{name}, a hard link that may share another file's contents" if stat.nlink > 1
        elsif !stat.directory?
          raise ConfigurationError, "#{dir} holds #{name}, which is not a regular file"
        end
      end

      def refuse_link(dir, root, name, path)
        target = File.realpath(path)

        raise ConfigurationError, "#{dir} holds #{name}, a link leading outside the directory" unless target.start_with?("#{root}/")
        raise ConfigurationError, "#{dir} holds #{name}, a link to a directory, which RPEFT cannot check" if File.directory?(target)
        raise ConfigurationError, "#{dir} holds #{name}, a link to something other than a regular file" unless File.file?(target)
      rescue Errno::ENOENT, Errno::ELOOP
        raise ConfigurationError, "#{dir} holds #{name}, a link that leads nowhere"
      end

      # Every JSON file is parsed rather than searched for the text auto_map, which a key can spell with
      # a Unicode escape for any of its letters; a multi-megabyte tokenizer.json takes tens of milliseconds.
      # NaN is allowed, as Python's json, which reads the configs, allows it.
      def refuse_auto_map(dir, name, path)
        # Configs and tokenizers run to tens of megabytes at most; a larger file would only be read into memory.
        if File.size(path) > MAX_JSON_BYTES
          raise ConfigurationError, "#{dir} holds #{name}, a JSON file larger than #{MAX_JSON_BYTES / 1024 / 1024} MB"
        end

        config = JSON.parse(File.read(path), allow_nan: true)
        return unless config.is_a?(Hash) && config.key?('auto_map')

        raise ConfigurationError, "#{dir} maps classes to custom code (auto_map in #{name}), which the GGUF converter could run"
      rescue JSON::ParserError => e
        raise ConfigurationError, "#{dir} holds #{name}, which is not valid JSON: #{e.message}"
      end

      def output_path(value)
        path = expanded_path(value, 'output')

        # File.exist? is false for a link whose target is missing, which the converter would write through.
        if File.exist?(path) || File.symlink?(path)
          raise ConfigurationError, "output #{path} already exists; remove it or choose another path"
        end
        raise ConfigurationError, "output directory #{File.dirname(path)} does not exist" unless File.directory?(File.dirname(path))

        path
      end

      def llama_cpp_directory(llama_cpp)
        dir = llama_cpp ? Arguments.path(llama_cpp, 'llama_cpp') : ENV.fetch('LLAMA_CPP_DIR', '')

        if dir.empty?
          raise DependencyError, 'GGUF export needs a llama.cpp checkout for its converter: clone ' \
                                 'https://github.com/ggml-org/llama.cpp and pass llama_cpp: (--llama-cpp) or set LLAMA_CPP_DIR'
        end

        expand(dir) || raise(ConfigurationError, "llama_cpp #{dir.inspect} cannot be expanded to a path")
      end

      def converter_path(llama_cpp_dir)
        converter = File.join(llama_cpp_dir, CONVERTER)
        raise DependencyError, "#{CONVERTER} not found in #{llama_cpp_dir}; is it a llama.cpp checkout?" unless File.file?(converter)

        converter
      end

      def metadata_editor_path(llama_cpp_dir)
        editor = File.join(llama_cpp_dir, METADATA_EDITOR)
        raise DependencyError, "#{METADATA_EDITOR} not found in #{llama_cpp_dir}, needed for chat_template: false" unless File.file?(editor)

        editor
      end

      def quantizer_path(llama_quantize, llama_cpp_dir)
        if llama_quantize
          path = expanded_path(llama_quantize, 'llama_quantize')
          return path if File.file?(path) && File.executable?(path)

          raise DependencyError, "#{path} is not an executable #{QUANTIZER}"
        end

        found = find_quantizer(llama_cpp_dir, ENV)
        return found if found

        raise DependencyError, "#{QUANTIZER} not found for the K-quant types; install llama.cpp (e.g. `brew install llama.cpp`), " \
                               'build it in the llama.cpp checkout, or pass llama_quantize: (--llama-quantize)'
      end

      def find_quantizer(llama_cpp_dir, env)
        candidates = env.fetch('PATH', '').split(File::PATH_SEPARATOR).reject(&:empty?).map { |dir| File.join(dir, QUANTIZER) }
        candidates << File.join(llama_cpp_dir, 'build', 'bin', QUANTIZER) if llama_cpp_dir

        candidates.find { |path| File.file?(path) && File.executable?(path) }
      end

      def expanded_path(value, name)
        path = Arguments.path(value, name)

        expand(path) || raise(ConfigurationError, "#{name} #{path.inspect} cannot be expanded to a path")
      end

      # File.expand_path raises for a ~user that does not exist; nil stands for that here.
      def expand(path)
        File.expand_path(path)
      rescue ArgumentError
        nil
      end

      def convert(model_dir, output, outtype, converter, env)
        run([python, converter, model_dir, '--outfile', output, '--outtype', outtype], CONVERTER, env)
      end

      # Ollama (checked with 0.33.1) wraps every prompt in the chat template a GGUF file carries, even when
      # a Modelfile gives a TEMPLATE of its own. A model fine-tuned on plain text expects its prompt as it was in
      # training, so for such a model the template is left out. The editor writes a whole new file, skips keys
      # the file does not have, and with --force skips its confirmation prompt.
      def remove_chat_template(staged, editor, env)
        edited = File.join(File.dirname(staged), 'model-no-chat-template.gguf')
        argv = [
          python, editor, '--remove-metadata', CHAT_TEMPLATE_KEY, '--remove-metadata', CHAT_TEMPLATE_NAMES_KEY, '--force', staged, edited
        ]

        run(argv, File.basename(editor), env)
        verify(edited, File.basename(editor))

        edited
      end

      # The converter loads some tokenizers with trust_remote_code=True. Besides the model directory, checked
      # by refuse_custom_code, that code could come from the Hugging Face Hub, which offline mode keeps out,
      # or from modules earlier runs left in the Hugging Face caches, which offline mode still loads. The caches
      # are therefore pointed at empty directories of this export; a local model needs nothing from them.
      def converter_env(dir)
        {
          'HF_HUB_OFFLINE' => '1',
          'TRANSFORMERS_OFFLINE' => '1',
          'HF_HOME' => File.join(dir, 'hf-home'),
          'HF_HUB_CACHE' => File.join(dir, 'hf-hub'),
          'HF_MODULES_CACHE' => File.join(dir, 'hf-modules'),
          'TRANSFORMERS_CACHE' => nil,
          'HUGGINGFACE_HUB_CACHE' => nil
        }
      end

      def run(argv, name, env = {})
        begin
          output, success = runner.call(argv, env)
        rescue SystemCallError => e
          raise DependencyError, "could not run #{name} (#{argv.first}): #{e.message}"
        end
        return if success

        tail = output.to_s.lines.last(20).join

        raise ExportError, "#{name} failed:\n#{tail}"
      end

      # The converter runs in its own process with the interpreter RPEFT uses, so it sees the same packages
      # without loading torch a second time into this process through PyCall.
      def python
        require_relative 'backend/python'

        Backend::Python.default_python || 'python3'
      end

      def verify(path, name)
        header = File.file?(path) ? File.binread(path, MAGIC.bytesize) : nil

        raise ExportError, "#{name} did not write a GGUF file" unless header == MAGIC
      end

      # A hard link puts the file at the output in one step and fails if anything is there by now, so a file
      # another process created since output_path checked is neither replaced nor removed.
      def publish(staged, output)
        File.link(staged, output)
      rescue Errno::EEXIST
        raise ConfigurationError, "output #{output} already exists; remove it or choose another path"
      rescue Errno::EPERM, Errno::ENOTSUP, Errno::EOPNOTSUPP
        copy_exclusively(staged, output)
      end

      # Some file systems, such as exFAT, have no hard links. Creating the output exclusively keeps the same
      # guarantee at the cost of a copy, and a file this creates is the only one a failed copy removes.
      def copy_exclusively(staged, output)
        created = false

        File.open(output, File::WRONLY | File::CREAT | File::EXCL | File::BINARY) do |file|
          created = true

          IO.copy_stream(staged, file)
        end
      rescue Errno::EEXIST
        raise ConfigurationError, "output #{output} already exists; remove it or choose another path"
      rescue StandardError
        File.delete(output) if created
        raise
      end
    end
  end
end
