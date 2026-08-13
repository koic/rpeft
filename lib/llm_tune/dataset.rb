# frozen_string_literal: true

require 'json'
require 'tmpdir'

module LLMTune
  # A training dataset. `Dataset.coerce` turns any value accepted by the `dataset:` keyword
  # (a path String, a Pathname, an Enumerable of record Hashes, or a Dataset) into a validated Dataset.
  #
  # Three JSONL line formats are accepted and detected from the first line: `{"text": ...}` is
  # passed through as is, `{"instruction": ..., "input": ..., "output": ...}` is rendered into
  # a text prompt, and `{"messages": [{"role": ..., "content": ...}, ...]}` is passed through
  # for the backend to render with the tokenizer's chat template.
  class Dataset
    def self.coerce(value)
      return value if value.is_a?(Dataset)
      return from_records(value) if value.is_a?(Enumerable)

      new(value)
    end

    # Builds a dataset from Ruby records (Hashes in any of the three formats, Symbol keys included)
    # by writing them to a temporary JSONL file, so in-memory data needs no intermediate file of its own.
    def self.from_records(records)
      path = File.join(Dir.mktmpdir('llm_tune-dataset-'), 'records.jsonl')
      written = 0

      File.open(path, 'w') do |out|
        records.each do |record|
          raise DatasetError, "each dataset record must be a Hash, got #{record.class}" unless record.is_a?(Hash)

          out.puts JSON.generate(record)
          written += 1
        end
      end
      raise DatasetError, 'dataset is empty: no records were given' if written.zero?

      new(path)
    end

    attr_reader :path, :format

    def initialize(source)
      source_path = existing_file(source)
      @format = detect_format(source_path)
      @path = @format == :instruction ? normalize(source_path) : source_path
    end

    # The format the backend sees. Instruction datasets are normalized to text here,
    # so only text and messages ever cross the boundary.
    def backend_format
      format == :messages ? 'messages' : 'text'
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
      valid = messages.is_a?(Array) && !messages.empty? && messages.all? { |message|
        message.is_a?(Hash) && message['role'].is_a?(String) && message['content'].is_a?(String)
      }

      return if valid

      raise DatasetError, "\"messages\" must be a non-empty array of objects with \"role\" and \"content\" strings: #{path}"
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

    # Rewrites a non-text dataset into text-format JSONL in
    # a temporary directory, so the backend only ever sees one format.
    def normalize(source_path)
      normalized = File.join(Dir.mktmpdir('llm_tune-dataset-'), File.basename(source_path))

      File.open(normalized, 'w') do |out|
        File.foreach(source_path).with_index(1) do |line, lineno|
          next if line.strip.empty?

          record = parse_line(line, source_path)
          out.puts JSON.generate('text' => render_instruction(record, lineno, source_path))
        end
      end

      normalized
    end

    def render_instruction(record, lineno, source_path)
      instruction = record['instruction']
      output = record['output']
      if !instruction.is_a?(String) || instruction.empty? || !output.is_a?(String)
        raise DatasetError, "line #{lineno}: instruction format requires \"instruction\" and \"output\" strings: #{source_path}"
      end

      input = record['input']
      text = "### Instruction:\n#{instruction}\n"
      text << "\n### Input:\n#{input}\n" if input && !input.to_s.empty?
      text << "\n### Response:\n#{output}"
      text
    end
  end
end
