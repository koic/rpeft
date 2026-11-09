# frozen_string_literal: true

require 'fileutils'
require 'json'
require 'tmpdir'

module RPEFT
  # A training dataset. `Dataset.coerce` turns any value accepted by the `dataset:` keyword
  # (a path String, a Pathname, an Enumerable of record Hashes, or a Dataset) into a validated Dataset.
  #
  # Three JSONL line formats are accepted and detected from the first line: `{"text": ...}` is
  # passed through as is, `{"instruction": ..., "input": ..., "output": ...}` is rendered into
  # a prompt and its completion, and `{"messages": [{"role": ..., "content": ...}, ...]}` is passed through
  # for the backend to render with the tokenizer's chat template.
  class Dataset
    BACKEND_FORMATS = {text: 'text', instruction: 'completion', messages: 'messages'}.freeze

    def self.coerce(value)
      return value if value.is_a?(Dataset)
      return from_records(value) if value.is_a?(Enumerable)

      new(Arguments.path(value, 'dataset'))
    end

    # Builds a dataset from Ruby records (Hashes in any of the three formats, Symbol keys included)
    # by writing them to a temporary JSONL file, so in-memory data needs no intermediate file of its own.
    def self.from_records(records)
      dir = Dir.mktmpdir('rpeft-dataset-')
      path = File.join(dir, 'records.jsonl')
      written = 0

      File.open(path, 'w') do |out|
        records.each do |record|
          raise DatasetError, "each dataset record must be a Hash, got #{record.class}" unless record.is_a?(Hash)

          out.puts JSON.generate(record)
          written += 1
        end
      end
      raise DatasetError, 'dataset is empty: no records were given' if written.zero?

      new(path, name: nil, temporary_dir: dir)
    rescue StandardError
      FileUtils.rm_rf(dir)
      raise
    end

    # name is the file name the dataset came from, recorded in the adapter metadata;
    # nil for in-memory records, which have no file of their own.
    attr_reader :path, :format, :name

    def initialize(source, name: File.basename(source.to_s), temporary_dir: nil)
      @name = name
      @temporary_dirs = [temporary_dir].compact
      source_path = existing_file(source)
      @format = detect_format(source_path)
      @path = @format == :instruction ? normalize(source_path) : source_path
    end

    # Removes the temporary copies this dataset wrote for the backend. The source file is left alone.
    def discard
      @temporary_dirs.each do |dir|
        FileUtils.rm_rf(dir)
      end
      @temporary_dirs.clear
    end

    # The format the backend sees. Instruction datasets are normalized to prompt and completion pairs here,
    # which keeps the boundary the backend needs to train on the completion alone.
    def backend_format
      BACKEND_FORMATS.fetch(format)
    end

    def messages?
      format == :messages
    end

    private

    def existing_file(source)
      path = File.expand_path(source)
      raise DatasetError, "dataset file not found: #{path}" unless File.file?(path)

      path
    end

    def detect_format(path)
      record = first_record(path)
      if record.is_a?(Hash)
        return :text if record.key?('text')
        return :instruction if record.key?('instruction') && record.key?('output')

        if record.key?('messages')
          validate_messages(record['messages'], path)
          return :messages
        end
      end

      raise DatasetError, "each dataset line must be a JSON object with a \"text\" key, \"instruction\"/\"output\" keys, or a \"messages\" key: #{path}"
    end

    def validate_messages(messages, path)
      return if Messages.valid?(messages)

      raise DatasetError, "\"messages\" must be a non-empty array of objects with a \"role\" of system, user, or assistant and a \"content\" string: #{path}"
    end

    def first_record(path)
      line = File.foreach(path).find { |l| !l.strip.empty? }
      raise DatasetError, "dataset is empty: #{path}" unless line

      parse_line(line, path)
    end

    def parse_line(line, path)
      JSON.parse(line)
    rescue JSON::ParserError => e
      raise DatasetError, "dataset is not JSONL (#{e.message}): #{path}"
    end

    # Rewrites an instruction dataset into prompt and completion JSONL in a temporary directory,
    # so the backend needs no knowledge of the instruction template.
    def normalize(source_path)
      dir = Dir.mktmpdir('rpeft-dataset-')
      @temporary_dirs << dir
      normalized = File.join(dir, File.basename(source_path))

      File.open(normalized, 'w') do |out|
        File.foreach(source_path).with_index(1) do |line, lineno|
          next if line.strip.empty?

          record = parse_line(line, source_path)
          prompt, completion = render_instruction(record, lineno, source_path)

          out.puts JSON.generate('prompt' => prompt, 'completion' => completion)
        end
      end

      normalized
    rescue StandardError
      discard
      raise
    end

    def render_instruction(record, lineno, source_path)
      instruction = record['instruction']
      output = record['output']
      if !instruction.is_a?(String) || instruction.empty? || !output.is_a?(String)
        raise DatasetError, "line #{lineno}: instruction format requires \"instruction\" and \"output\" strings: #{source_path}"
      end

      input = record['input']
      prompt = "### Instruction:\n#{instruction}\n"
      prompt << "\n### Input:\n#{input}\n" if input && !input.to_s.empty?
      prompt << "\n### Response:\n"

      [prompt, output]
    end
  end
end
