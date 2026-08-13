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
      Generation.run(
        model: path,
        prompt: prompt,
        max_new_tokens: max_new_tokens,
        device: device,
        temperature: temperature,
        top_p: top_p,
        seed: seed
      )
    end
  end
end
