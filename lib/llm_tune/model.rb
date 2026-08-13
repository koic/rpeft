# frozen_string_literal: true

module LLMTune
  # A full model: either a local directory (typically the result of merging a LoRA adapter)
  # or a Hugging Face Hub model id.
  class Model
    REPLACE_MARKERS = %w(config.json).freeze

    attr_reader :path, :base_model

    def initialize(path:, base_model: nil)
      # A Hub id such as "Qwen/Qwen2.5-Coder-0.5B" must stay verbatim; only real local directories are expanded.
      @path = File.directory?(path) ? File.expand_path(path) : path.to_s
      @base_model = base_model
    end

    def save(dest)
      @path = Directory.move(@path, dest, replace_markers: REPLACE_MARKERS)
      self
    end

    # Generates a completion for the prompt with greedy decoding, so the output is deterministic
    # for a given model and prompt.
    def generate(prompt, max_new_tokens: 64, device: :auto)
      if !max_new_tokens.is_a?(Integer) || !max_new_tokens.positive?
        raise ConfigurationError, "max_new_tokens must be a positive Integer, got #{max_new_tokens.inspect}"
      end

      result = LLMTune.backend.generate(
        'model' => path,
        'prompt' => prompt,
        'max_new_tokens' => max_new_tokens,
        'device' => Device.resolve(device).to_s,
        'trust_remote_code' => false
      )
      result['completion']
    end
  end
end
