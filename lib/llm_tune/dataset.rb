# frozen_string_literal: true

require 'json'

module LLMTune
  # A training dataset: a JSONL file whose lines are JSON objects with a "text" key.
  # `Dataset.coerce` turns any value accepted by the `dataset:` keyword (a path String,
  # a Pathname, or a Dataset) into a validated Dataset.
  class Dataset
    def self.coerce(value)
      value.is_a?(Dataset) ? value : new(value)
    end

    attr_reader :path

    def initialize(source)
      @path = validate_file(source)
    end

    private

    def validate_file(source)
      path = File.expand_path(source)
      raise DatasetError, "dataset file not found: #{path}" unless File.file?(path)

      first_line = File.foreach(path).find { |line| !line.strip.empty? }
      raise DatasetError, "dataset is empty: #{path}" unless first_line

      validate_line(first_line, path)

      path
    end

    def validate_line(line, path)
      record = JSON.parse(line)
      return if record.is_a?(Hash) && record.key?('text')

      raise DatasetError, %(each dataset line must be a JSON object with a "text" key: #{path})
    rescue JSON::ParserError => e
      raise DatasetError, "dataset is not JSONL (#{e.message}): #{path}"
    end
  end
end
