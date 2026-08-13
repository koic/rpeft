# frozen_string_literal: true

module LLMTune
  # @api private
  #
  # Shared request building for text generation. Model#generate runs
  # a standalone model; Adapter#generate adds adapter_path so the backend
  # applies the adapter on top of the base model without merging.
  module Generation
    module_function

    def run(model:, prompt:, max_new_tokens: 64, device: :auto, temperature: nil, top_p: nil, seed: nil, adapter_path: nil)
      validate(max_new_tokens, temperature, top_p, seed)

      request = {
        'model' => model,
        'prompt' => prompt,
        'max_new_tokens' => max_new_tokens,
        'device' => Device.resolve(device).to_s,
        'trust_remote_code' => false
      }
      request['adapter_path'] = adapter_path if adapter_path
      request['temperature'] = temperature if temperature
      request['top_p'] = top_p if top_p
      request['seed'] = seed if seed

      LLMTune.backend.generate(request)['completion']
    end

    def validate(max_new_tokens, temperature, top_p, seed)
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
