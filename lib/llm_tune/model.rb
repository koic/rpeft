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

    # Generates a completion for the prompt. Decoding is greedy (and thus deterministic for a given model and prompt)
    # unless temperature or top_p turns sampling on.
    def generate(prompt, max_new_tokens: 64, device: :auto, temperature: nil, top_p: nil, seed: nil)
      validate_generation_options(max_new_tokens, temperature, top_p, seed)

      request = {
        'model' => path,
        'prompt' => prompt,
        'max_new_tokens' => max_new_tokens,
        'device' => Device.resolve(device).to_s,
        'trust_remote_code' => false
      }
      request['temperature'] = temperature if temperature
      request['top_p'] = top_p if top_p
      request['seed'] = seed if seed

      LLMTune.backend.generate(request)['completion']
    end

    private

    def validate_generation_options(max_new_tokens, temperature, top_p, seed)
      if !max_new_tokens.is_a?(Integer) || !max_new_tokens.positive?
        raise ConfigurationError, "max_new_tokens must be a positive Integer, got #{max_new_tokens.inspect}"
      end

      if temperature && !(temperature.is_a?(Numeric) && temperature.positive?)
        raise ConfigurationError, "temperature must be a positive number, got #{temperature.inspect}"
      end

      if top_p && !(top_p.is_a?(Numeric) && top_p.positive? && top_p <= 1)
        raise ConfigurationError, "top_p must be within (0.0, 1.0], got #{top_p.inspect}"
      end
      return if seed.nil? || seed.is_a?(Integer)

      raise ConfigurationError, "seed must be an Integer, got #{seed.inspect}"
    end
  end
end
