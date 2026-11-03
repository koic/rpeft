# frozen_string_literal: true

module RPEFT
  class Error < StandardError; end

  # Python, PyCall, or a required Python package is missing or unusable.
  class DependencyError < Error; end

  # Invalid options were passed to the Ruby API.
  class ConfigurationError < Error; end

  # The dataset file is missing or not in the expected JSONL format.
  class DatasetError < Error; end

  # The base model or an adapter directory could not be loaded.
  class ModelLoadError < Error; end

  # LoRA training failed. The original Python error is available via #cause.
  class TrainingError < Error; end

  # Merging the adapter into the base model failed.
  class MergeError < Error; end

  # Text generation failed.
  class GenerationError < Error; end
end
