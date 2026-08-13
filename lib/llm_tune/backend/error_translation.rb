# frozen_string_literal: true

module LLMTune
  module Backend
    # Maps the "[stage:xxx]" token that the bundled Python backend prefixes to its error messages
    # onto the Ruby exception hierarchy.
    # Parsing arbitrary Python exception class names would be brittle across library versions;
    # parsing a sentinel we emit ourselves is deterministic.
    module ErrorTranslation
      STAGE_MAP = {
        'dependency' => DependencyError,
        'device' => DependencyError,
        'model_load' => ModelLoadError,
        'dataset' => DatasetError,
        'training' => TrainingError,
        'adapter_save' => TrainingError,
        'merge' => MergeError,
        'generation' => GenerationError
      }.freeze

      def self.translate(error, default:)
        stage = error.message[/\[stage:(\w+)\]/, 1]
        error_class = STAGE_MAP.fetch(stage, default)
        error_class.new(error.message.sub(/\[stage:\w+\]\s*/, ''))
      end
    end
  end
end
