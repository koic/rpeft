# frozen_string_literal: true

require 'json'
require 'tmpdir'

module LLMTune
  # A training dataset. `Dataset.coerce` turns any value accepted by the `dataset:` keyword
  # (a path String, a Pathname, or a Dataset) into a validated Dataset.
  #
  # Two JSONL line formats are accepted and detected from the first line: `{"text": ...}` is
  # passed through as is, and `{"instruction": ..., "input": ..., "output": ...}` is rendered
  # into a text prompt, so the backend always consumes text-format JSONL.
  class Dataset
    def self.coerce(value)
      value.is_a?(Dataset) ? value : new(value)
    end

    attr_reader :path, :format

    def initialize(source)
      source_path = existing_file(source)
      @format = detect_format(source_path)
      @path = @format == :text ? source_path : normalize(source_path)
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
      end

      raise DatasetError, "each dataset line must be a JSON object with a \"text\" key or \"instruction\"/\"output\" keys: #{path}"
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
