# frozen_string_literal: true

require 'test_helper'
require 'minitest/mock'

class TestGGUF < RPEFTTest
  def setup
    super
    @workdir = Dir.mktmpdir('rpeft_test')
    @model_dir = File.join(@workdir, 'model')
    FileUtils.mkdir_p(@model_dir)
    File.write(File.join(@model_dir, 'config.json'), %({"model_type":"qwen2"}))
    File.write(File.join(@model_dir, 'tokenizer_config.json'), %({"model_max_length":32768}))

    @llama_cpp = File.join(@workdir, 'llama.cpp')
    FileUtils.mkdir_p(@llama_cpp)
    @converter = File.join(@llama_cpp, 'convert_hf_to_gguf.py')
    File.write(@converter, "# converter\n")

    @output = File.join(@workdir, 'model.gguf')
    @commands = []
    @envs = []
    RPEFT::GGUF.runner = method(:fake_run)

    @env = save_env('LLAMA_CPP_DIR', 'PATH', 'HOME')
    ENV.delete('LLAMA_CPP_DIR')
    ENV['PATH'] = File.join(@workdir, 'empty-bin')
  end

  def teardown
    RPEFT::GGUF.runner = nil
    restore_env(@env)
    FileUtils.remove_entry(@workdir)
    super
  end

  def test_converts_to_q8_0_by_default
    path = export

    assert_equal @output, path
    assert_equal [python, @converter, @model_dir, '--outfile'], @commands.first.first(4)
    assert_equal %w(--outtype q8_0), @commands.first.last(2)
    assert_equal 'GGUF', File.binread(@output, 4)
  end

  def test_runs_the_converter_offline_with_caches_of_its_own
    export

    env = @envs.first
    outfile = @commands.first[@commands.first.index('--outfile') + 1]

    assert_equal '1', env['HF_HUB_OFFLINE']
    assert_equal '1', env['TRANSFORMERS_OFFLINE']
    # Modules an earlier run cached would load even offline, so the converter sees none of the caches.
    %w(HF_HOME HF_HUB_CACHE HF_MODULES_CACHE).each do |key|
      assert_equal File.dirname(outfile), File.dirname(env[key]), key
    end
    assert env.key?('TRANSFORMERS_CACHE')
    assert_nil env['TRANSFORMERS_CACHE']
  end

  def test_converts_the_16_bit_types_in_one_step
    export(quantize: :bf16)

    assert_equal 'bf16', @commands.last.last
    assert_equal 1, @commands.size
  end

  def test_leaves_nothing_but_the_output_behind
    export

    assert_equal %w(llama.cpp model model.gguf), Dir.children(@workdir).sort
  end

  def test_keeps_the_chat_template_by_default
    install_metadata_editor

    export

    assert_equal 1, @commands.size
  end

  def test_leaves_out_the_chat_template_when_asked
    editor = install_metadata_editor

    export(chat_template: false)

    convert, edit = @commands
    converted = convert[convert.index('--outfile') + 1]

    assert_equal [
      python, editor, '--remove-metadata', 'tokenizer.chat_template', '--remove-metadata', 'tokenizer.chat_templates',
      '--force', converted
    ], edit[0..7]
    assert_equal File.dirname(converted), File.dirname(edit.last)
    assert_equal 'GGUF', File.binread(@output, 4)
    assert_equal %w(llama.cpp model model.gguf), Dir.children(@workdir).sort
  end

  def test_leaves_out_the_chat_template_after_quantizing
    install_metadata_editor
    ENV['PATH'] = File.dirname(install_quantizer(File.join(@workdir, 'bin')))

    export(quantize: :q4_k_m, chat_template: false)

    _convert, quantize, edit = @commands

    assert_equal quantize[2], edit[-2]
  end

  def test_refuses_to_leave_out_the_chat_template_without_the_metadata_editor
    error = assert_raises(RPEFT::DependencyError) { export(chat_template: false) }

    assert_match(/gguf_new_metadata\.py not found/, error.message)
    assert_empty @commands
  end

  def test_rejects_a_chat_template_option_that_is_not_true_or_false
    error = assert_raises(RPEFT::ConfigurationError) { export(chat_template: 'no') }

    assert_match(/chat_template must be true or false/, error.message)
    assert_empty @commands
  end

  def test_names_the_metadata_editor_when_it_writes_no_gguf_file
    install_metadata_editor
    RPEFT::GGUF.runner = lambda do |argv, _env|
      if argv.include?('--remove-metadata')
        File.write(argv.last, 'not gguf')
      else
        File.binwrite(argv[argv.index('--outfile') + 1], 'GGUF')
      end
      ['', true]
    end

    error = assert_raises(RPEFT::ExportError) { export(chat_template: false) }

    assert_match(/gguf_new_metadata\.py did not write a GGUF file/, error.message)
    refute File.exist?(@output)
  end

  def test_never_hands_the_requested_name_to_the_converter
    # The converter would expand {ftype} into the type and write a file under another name.
    output = File.join(@workdir, 'model.{ftype}.gguf')

    export(output: output)

    outfile = @commands.first[@commands.first.index('--outfile') + 1]

    refute_includes outfile, '{ftype}'
    assert_path_exists output
  end

  def test_quantizes_the_k_types_from_a_bf16_intermediate_next_to_the_output_that_is_removed_afterwards
    quantizer = install_quantizer(File.join(@workdir, 'bin'))
    ENV['PATH'] = File.dirname(quantizer)

    export(quantize: :q4_k_m)

    convert, quantize = @commands
    intermediate = convert[convert.index('--outfile') + 1]

    assert_equal 'bf16', convert.last
    assert_equal [quantizer, intermediate, quantize[2], 'Q4_K_M'], quantize
    assert_equal @workdir, File.dirname(File.dirname(intermediate))
    refute File.exist?(intermediate)
    assert_equal 'GGUF', File.binread(@output, 4)
  end

  def test_accepts_the_type_as_a_string_in_any_case
    quantizer = install_quantizer(File.join(@workdir, 'bin'))
    ENV['PATH'] = File.dirname(quantizer)

    export(quantize: 'Q5_K_M')

    assert_equal 'Q5_K_M', @commands.last.last
  end

  def test_finds_llama_quantize_in_the_llama_cpp_build
    quantizer = install_quantizer(File.join(@llama_cpp, 'build', 'bin'))

    export(quantize: :q6_k)

    assert_equal quantizer, @commands.last.first
  end

  def test_uses_an_explicit_llama_quantize
    quantizer = install_quantizer(File.join(@workdir, 'tools'))

    export(quantize: :q4_k_m, llama_quantize: quantizer)

    assert_equal quantizer, @commands.last.first
  end

  def test_refuses_the_k_types_without_llama_quantize
    error = assert_raises(RPEFT::DependencyError) { export(quantize: :q4_k_m) }

    assert_match(/llama-quantize/, error.message)
    assert_empty @commands
  end

  def test_takes_the_llama_cpp_checkout_from_the_environment
    ENV['LLAMA_CPP_DIR'] = @llama_cpp

    RPEFT::GGUF.export(@model_dir, @output)

    assert_equal @converter, @commands.first[1]
  end

  def test_refuses_to_run_without_a_llama_cpp_checkout
    error = assert_raises(RPEFT::DependencyError) { RPEFT::GGUF.export(@model_dir, @output) }

    assert_match(/LLAMA_CPP_DIR/, error.message)
    assert_empty @commands
  end

  def test_refuses_a_directory_without_the_converter
    FileUtils.rm(@converter)

    error = assert_raises(RPEFT::DependencyError) { export }

    assert_match(/convert_hf_to_gguf.py not found/, error.message)
  end

  def test_expands_a_home_relative_model_path
    ENV['HOME'] = @workdir

    export(model: '~/model')

    assert_equal @model_dir, @commands.first[2]
  end

  def test_rejects_a_path_with_an_unknown_user
    error = assert_raises(RPEFT::ConfigurationError) { export(model: '~rpeft_no_such_user/model') }

    assert_match(/cannot be expanded/, error.message)
  end

  def test_reports_a_tool_that_cannot_be_started
    RPEFT::GGUF.runner = ->(argv, _env) { raise Errno::ENOENT, argv.first }

    error = assert_raises(RPEFT::DependencyError) { export }

    assert_match(/could not run convert_hf_to_gguf.py/, error.message)
  end

  def test_rejects_an_unknown_type
    error = assert_raises(RPEFT::ConfigurationError) { export(quantize: :q2_k) }

    assert_match(/quantize must be one of/, error.message)
  end

  def test_rejects_a_hub_model_id
    error = assert_raises(RPEFT::ConfigurationError) {
      RPEFT::GGUF.export('Qwen/Qwen2.5-Coder-0.5B', @output, llama_cpp: @llama_cpp)
    }

    assert_match(/local model directory/, error.message)
  end

  def test_rejects_an_adapter_directory
    File.write(File.join(@model_dir, 'adapter_config.json'), '{}')

    error = assert_raises(RPEFT::ConfigurationError) { export }

    assert_match(/merge it/, error.message)
  end

  def test_rejects_a_directory_without_a_config_json
    FileUtils.rm(File.join(@model_dir, 'config.json'))

    assert_raises(RPEFT::ConfigurationError) { export }
  end

  def test_rejects_a_model_shipping_python_code
    File.write(File.join(@model_dir, 'tokenization_custom.py'), "import os\n")

    error = assert_raises(RPEFT::ConfigurationError) { export }

    assert_match(/Python code/, error.message)
    assert_empty @commands
  end

  def test_rejects_python_code_in_a_hidden_directory
    FileUtils.mkdir_p(File.join(@model_dir, '.hidden'))
    File.write(File.join(@model_dir, '.hidden', 'module.py'), "import os\n")

    error = assert_raises(RPEFT::ConfigurationError) { export }

    assert_match(%r|Python code \(\.hidden/module\.py\)|, error.message)
    assert_empty @commands
  end

  def test_rejects_a_link_to_a_directory_it_cannot_check
    FileUtils.mkdir_p(File.join(@model_dir, 'inside'))
    File.symlink(File.join(@model_dir, 'inside'), File.join(@model_dir, 'linked'))

    error = assert_raises(RPEFT::ConfigurationError) { export }

    assert_match(/linked, a link to a directory/, error.message)
  end

  def test_rejects_python_code_with_an_upper_case_extension
    # A case-insensitive file system, such as macOS's default one, serves this for configuration_x.py.
    File.write(File.join(@model_dir, 'configuration_x.PY'), "import os\n")

    error = assert_raises(RPEFT::ConfigurationError) { export }

    assert_match(/Python code \(configuration_x\.PY\)/, error.message)
    assert_empty @commands
  end

  def test_rejects_auto_map_in_a_json_file_with_an_upper_case_extension
    File.write(File.join(@model_dir, 'extra.JSON'), %({"auto_map":{"AutoConfig":"a.B"}}))

    error = assert_raises(RPEFT::ConfigurationError) { export }

    assert_match(/auto_map in extra\.JSON/, error.message)
  end

  def test_rejects_a_link_leading_outside_the_directory
    secret = File.join(@workdir, 'credentials')
    File.write(secret, 'secret')
    File.symlink(secret, File.join(@model_dir, 'chat_template.jinja'))

    error = assert_raises(RPEFT::ConfigurationError) { export }

    assert_match(/chat_template\.jinja, a link leading outside the directory/, error.message)
    assert_empty @commands
  end

  def test_accepts_a_link_to_a_file_inside_the_directory
    File.write(File.join(@model_dir, 'template.jinja'), '{{ messages }}')
    File.symlink(File.join(@model_dir, 'template.jinja'), File.join(@model_dir, 'chat_template.jinja'))

    export

    assert_equal 'GGUF', File.binread(@output, 4)
  end

  def test_rejects_a_link_that_leads_nowhere
    File.symlink(File.join(@model_dir, 'missing'), File.join(@model_dir, 'chat_template.jinja'))

    error = assert_raises(RPEFT::ConfigurationError) { export }

    assert_match(/a link that leads nowhere/, error.message)
  end

  def test_rejects_a_hard_link_that_may_share_another_files_contents
    secret = File.join(@workdir, 'credentials')
    File.write(secret, 'secret')
    File.link(secret, File.join(@model_dir, 'chat_template.jinja'))

    error = assert_raises(RPEFT::ConfigurationError) { export }

    assert_match(/chat_template\.jinja, a hard link/, error.message)
  end

  def test_rejects_a_special_file_that_could_stall_the_converter
    File.mkfifo(File.join(@model_dir, 'tokenizer.json'))

    error = assert_raises(RPEFT::ConfigurationError) { export }

    assert_match(/tokenizer\.json, which is not a regular file/, error.message)
    assert_empty @commands
  end

  def test_accepts_a_directory_whose_name_ends_in_json
    FileUtils.mkdir_p(File.join(@model_dir, 'extra.json'))

    export

    assert_equal 'GGUF', File.binread(@output, 4)
  end

  def test_rejects_a_json_file_too_large_to_be_a_config
    size = File.method(:size)
    limit = RPEFT::GGUF::MAX_JSON_BYTES
    File.write(File.join(@model_dir, 'huge.json'), '{}')

    error = File.stub(:size, ->(path) { path.end_with?('huge.json') ? limit + 1 : size.call(path) }) {
      assert_raises(RPEFT::ConfigurationError) { export }
    }

    assert_match(/huge\.json, a JSON file larger than 256 MB/, error.message)
  end

  def test_exports_a_model_directory_reached_through_a_link
    linked = File.join(@workdir, 'linked-model')
    File.symlink(@model_dir, linked)

    export(model: linked)

    assert_equal 'GGUF', File.binread(@output, 4)
  end

  def test_rejects_a_model_mapping_classes_to_custom_code
    File.write(File.join(@model_dir, 'tokenizer_config.json'), %({"auto_map":{"AutoTokenizer":["a.B",null]}}))

    error = assert_raises(RPEFT::ConfigurationError) { export }

    assert_match(/auto_map/, error.message)
    assert_empty @commands
  end

  def test_rejects_auto_map_spelled_with_an_escape
    File.write(File.join(@model_dir, 'config.json'), %({"model_type":"qwen2","auto\\u005fmap":{"AutoConfig":"a.B"}}))

    error = assert_raises(RPEFT::ConfigurationError) { export }

    assert_match(/auto_map in config\.json/, error.message)
    assert_empty @commands
  end

  def test_rejects_auto_map_in_a_config_holding_nan_as_python_reads_it
    File.write(File.join(@model_dir, 'config.json'), %({"model_type":"qwen2","rope_scaling":NaN,"auto_map":{"AutoConfig":"a.B"}}))

    error = assert_raises(RPEFT::ConfigurationError) { export }

    assert_match(/auto_map in config\.json/, error.message)
  end

  def test_rejects_custom_code_mapped_in_a_config_another_config_names
    File.write(File.join(@model_dir, 'config.json'), %({"model_type":"qwen2","configuration_files":["config.4.0.0.json"]}))
    File.write(File.join(@model_dir, 'config.4.0.0.json'), %({"auto_map":{"AutoConfig":"org/repo--module.Config"}}))

    error = assert_raises(RPEFT::ConfigurationError) { export }

    assert_match(/auto_map in config\.4\.0\.0\.json/, error.message)
    assert_empty @commands
  end

  def test_copies_the_file_to_the_output_where_hard_links_are_unavailable
    File.stub(:link, ->(*) { raise Errno::EPERM }) do
      export
    end

    assert_equal "GGUF\x03\x00\x00\x00".b, File.binread(@output)
    assert_empty Dir.glob('.rpeft-gguf-*', base: @workdir)
  end

  def test_keeps_a_file_written_at_the_output_meanwhile_where_hard_links_are_unavailable
    RPEFT::GGUF.runner = lambda do |argv, _env|
      File.write(@output, 'written meanwhile')
      File.binwrite(argv[argv.index('--outfile') + 1], 'GGUF')
      ['', true]
    end

    File.stub(:link, ->(*) { raise Errno::EPERM }) do
      assert_raises(RPEFT::ConfigurationError) { export }
    end
    assert_equal 'written meanwhile', File.read(@output)
  end

  def test_removes_a_partial_copy_where_hard_links_are_unavailable
    File.stub(:link, ->(*) { raise Errno::EPERM }) do
      IO.stub(:copy_stream, ->(*) { raise Errno::ENOSPC }) do
        assert_raises(RPEFT::ExportError) { export }
      end
    end
    refute File.exist?(@output)
  end

  def test_names_the_quantizer_when_it_writes_no_gguf_file
    quantizer = install_quantizer(File.join(@workdir, 'bin'))
    RPEFT::GGUF.runner = lambda do |argv, env|
      next fake_run(argv, env) unless argv.first == quantizer

      File.write(argv[2], 'not gguf')
      ['', true]
    end

    error = assert_raises(RPEFT::ExportError) { export(quantize: :q4_k_m, llama_quantize: quantizer) }

    assert_match(/llama-quantize did not write a GGUF file/, error.message)
  end

  def test_refuses_to_overwrite_an_existing_file
    File.write(@output, 'keep me')

    error = assert_raises(RPEFT::ConfigurationError) { export }

    assert_match(/already exists/, error.message)
    assert_equal 'keep me', File.read(@output)
  end

  def test_refuses_a_link_at_the_output_even_when_its_target_is_missing
    File.symlink(File.join(@workdir, 'missing-target'), @output)

    error = assert_raises(RPEFT::ConfigurationError) { export }

    assert_match(/already exists/, error.message)
    refute File.exist?(File.join(@workdir, 'missing-target'))
  end

  def test_keeps_a_file_another_process_wrote_at_the_output_during_the_export
    RPEFT::GGUF.runner = lambda do |argv, _env|
      File.write(@output, 'written meanwhile')
      File.binwrite(argv[argv.index('--outfile') + 1], 'GGUF')
      ['', true]
    end

    assert_raises(RPEFT::ConfigurationError) do
      export
    end
    assert_equal 'written meanwhile', File.read(@output)
    assert_empty Dir.glob('.rpeft-gguf-*', base: @workdir)
  end

  def test_refuses_an_output_in_a_missing_directory
    assert_raises(RPEFT::ConfigurationError) { export(output: File.join(@workdir, 'missing', 'model.gguf')) }
  end

  def test_reports_a_failing_converter_and_removes_its_partial_output
    RPEFT::GGUF.runner = lambda do |argv, _env|
      File.write(argv[argv.index('--outfile') + 1], 'partial')
      ["Loading model\nNotImplementedError: Architecture 'Foo' not supported!\n", false]
    end

    error = assert_raises(RPEFT::ExportError) { export }

    assert_match(/Architecture 'Foo' not supported/, error.message)
    refute File.exist?(@output)
    assert_empty Dir.glob('.rpeft-gguf-*', base: @workdir)
  end

  def test_reports_a_failing_quantizer_and_removes_the_intermediate
    quantizer = install_quantizer(File.join(@workdir, 'bin'))
    RPEFT::GGUF.runner = lambda do |argv, env|
      next fake_run(argv, env) unless argv.first == quantizer

      File.write(argv[2], 'partial')
      ["main: invalid ftype\n", false]
    end

    error = assert_raises(RPEFT::ExportError) { export(quantize: :q4_k_m, llama_quantize: quantizer) }

    assert_match(/llama-quantize failed:\nmain: invalid ftype/, error.message)
    refute File.exist?(@output)
    assert_empty Dir.glob('.rpeft-gguf-*', base: @workdir)
  end

  def test_rejects_an_output_that_is_not_gguf
    RPEFT::GGUF.runner = lambda do |argv, _env|
      File.write(argv[argv.index('--outfile') + 1], 'not gguf')
      ['', true]
    end

    assert_raises(RPEFT::ExportError) do
      export
    end
    refute File.exist?(@output)
  end

  def test_model_exports_its_own_directory
    path = RPEFT::Model.new(path: @model_dir).export_gguf(@output, llama_cpp: @llama_cpp)

    assert_equal @output, path
    assert_equal @model_dir, @commands.first[2]
  end

  def test_model_passes_the_chat_template_option_on
    install_metadata_editor

    RPEFT::Model.new(path: @model_dir).export_gguf(@output, chat_template: false, llama_cpp: @llama_cpp)

    assert_includes @commands.last, '--remove-metadata'
  end

  def test_tools_reports_what_would_be_used
    quantizer = install_quantizer(File.join(@llama_cpp, 'build', 'bin'))

    tools = RPEFT::GGUF.tools(env: {'LLAMA_CPP_DIR' => @llama_cpp, 'PATH' => ''})

    assert_equal @llama_cpp, tools['llama_cpp_dir']
    assert_equal @converter, tools['converter']
    assert_equal quantizer, tools['quantizer']
  end

  def test_tools_reports_nothing_without_llama_cpp
    tools = RPEFT::GGUF.tools(env: {'PATH' => ''})

    assert_nil tools['llama_cpp_dir']
    assert_nil tools['converter']
    assert_nil tools['quantizer']
  end

  def test_tools_never_raises_for_a_llama_cpp_dir_that_cannot_be_expanded
    tools = RPEFT::GGUF.tools(env: {'LLAMA_CPP_DIR' => '~rpeft_no_such_user/llama.cpp', 'PATH' => ''})

    assert_equal '~rpeft_no_such_user/llama.cpp', tools['llama_cpp_dir']
    assert_nil tools['converter']
  end

  private

  def export(model: @model_dir, output: @output, **options)
    RPEFT::GGUF.export(model, output, llama_cpp: @llama_cpp, **options)
  end

  def python
    RPEFT::Backend::Python.default_python || 'python3'
  end

  def install_metadata_editor
    path = File.join(@llama_cpp, RPEFT::GGUF::METADATA_EDITOR)
    FileUtils.mkdir_p(File.dirname(path))
    File.write(path, "# editor\n")
    path
  end

  def fake_run(argv, env)
    @commands << argv
    @envs << env
    output = if argv.include?('--outfile')
      argv[argv.index('--outfile') + 1]
    elsif argv.include?('--remove-metadata')
      argv.last
    else
      argv[2]
    end
    File.binwrite(output, "GGUF\x03\x00\x00\x00")
    ['', true]
  end

  def install_quantizer(dir)
    FileUtils.mkdir_p(dir)
    path = File.join(dir, 'llama-quantize')
    File.write(path, "#!/bin/sh\n")
    File.chmod(0o755, path)
    path
  end
end
