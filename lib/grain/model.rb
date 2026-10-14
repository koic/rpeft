# frozen_string_literal: true

require 'json'
require 'time'

module Grain
  # A full model: either a local directory (typically the result of merging a LoRA adapter)
  # or a Hugging Face Hub model id.
  class Model
    METADATA_FILE = 'grain.json'
    # Not config.json: it is a common name outside model directories, and a mistyped save destination
    # holding one would otherwise be deleted.
    REPLACE_MARKERS = [METADATA_FILE].freeze

    attr_reader :path, :base_model

    # Marks dir as a merged model saved by Grain, which is what lets a later #save replace it.
    def self.write_metadata(dir, base_model:)
      metadata = {
        'schema_version' => 1,
        'grain_version' => VERSION,
        'kind' => 'merged_model',
        'base_model' => base_model,
        'created_at' => Time.now.iso8601
      }
      File.write(File.join(dir, METADATA_FILE), "#{JSON.pretty_generate(metadata)}\n")
    end

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

    # Continues a chat transcript and returns the assistant's reply. The transcript is
    # rendered with the model's own chat template, which is what a model fine-tuned on
    # chat-format data expects to see.
    def chat(messages, max_new_tokens: 64, device: :auto, temperature: nil, top_p: nil, seed: nil)
      Generation.run(
        model: path,
        messages: messages,
        max_new_tokens: max_new_tokens,
        device: device,
        temperature: temperature,
        top_p: top_p,
        seed: seed
      )
    end
  end
end
