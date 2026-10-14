# frozen_string_literal: true

require 'fileutils'
require 'json'
require 'tmpdir'

module Grain
  module LoRA
    # A trained LoRA adapter directory: the learned B A matrices plus the configuration
    # and metadata needed to merge them back into the base model.
    class Adapter
      METADATA_FILE = 'grain.json'
      PEFT_CONFIG_FILE = 'adapter_config.json'
      REPLACE_MARKERS = [METADATA_FILE, PEFT_CONFIG_FILE].freeze

      attr_reader :path, :base_model, :rank, :alpha, :dropout, :target_modules, :metadata

      class << self
        def load(path, base_model: nil)
          dir = File.expand_path(path)
          metadata_file = File.join(dir, METADATA_FILE)
          peft_file = File.join(dir, PEFT_CONFIG_FILE)

          if File.file?(metadata_file)
            from_metadata(dir, parse_json(metadata_file), base_model)
          elsif File.file?(peft_file)
            from_peft_config(dir, parse_json(peft_file), base_model)
          else
            raise ModelLoadError, "#{dir} is not a LoRA adapter directory (no #{METADATA_FILE} or #{PEFT_CONFIG_FILE} found)"
          end
        end

        private

        def from_metadata(dir, metadata, base_model)
          lora = metadata['lora'] || {}
          new(
            path: dir,
            base_model: base_model || metadata['base_model'],
            rank: lora['rank'],
            alpha: lora['alpha'],
            dropout: lora['dropout'],
            target_modules: lora['target_modules'],
            metadata: metadata
          )
        end

        # Fallback for adapter directories produced by plain PEFT,
        # which have no grain.json.
        def from_peft_config(dir, config, base_model)
          new(
            path: dir,
            base_model: base_model || config['base_model_name_or_path'],
            rank: config['r'],
            alpha: config['lora_alpha'],
            dropout: config['lora_dropout'],
            target_modules: config['target_modules'],
            metadata: {}
          )
        end

        def parse_json(file)
          JSON.parse(File.read(file))
        rescue JSON::ParserError => e
          raise ModelLoadError, "failed to parse #{file}: #{e.message}"
        end
      end

      def initialize(path:, base_model:, rank: nil, alpha: nil, dropout: nil, target_modules: nil, metadata: {})
        @path = File.expand_path(path)
        @base_model = base_model
        @rank = rank
        @alpha = alpha
        @dropout = dropout
        @target_modules = target_modules
        @metadata = metadata.freeze
      end

      def save(dest)
        @path = Directory.move(@path, dest, replace_markers: REPLACE_MARKERS)
        self
      end

      # Merges the learned B A update into the base model weights
      # (W = W0 + B A) and returns the resulting standalone Model.
      def merge(output_dir: nil)
        unless base_model
          raise MergeError, 'base model is unknown; pass base_model: to Adapter.load ' \
                            '(or --model to the CLI)'
        end

        merged_dir = File.expand_path(output_dir || Dir.mktmpdir('grain-merged-'))
        result = Grain.backend.merge_lora(
          'adapter_path' => path,
          'output_dir' => merged_dir,
          'base_model' => base_model
        )

        model_dir = result['model_path'] || merged_dir
        merged_base_model = result['base_model'] || base_model
        FileUtils.mkdir_p(model_dir)
        Model.write_metadata(model_dir, base_model: merged_base_model)

        Model.new(path: model_dir, base_model: merged_base_model)
      end

      # Generates with the adapter applied on top of the base model, without merging.
      # Useful for verifying an adapter before producing a merged model.
      def generate(prompt, max_new_tokens: 64, device: :auto, temperature: nil, top_p: nil, seed: nil)
        require_base_model

        Generation.run(
          model: base_model,
          prompt: prompt,
          adapter_path: path,
          max_new_tokens: max_new_tokens,
          device: device,
          temperature: temperature,
          top_p: top_p,
          seed: seed
        )
      end

      # Continues a chat transcript with the adapter applied on top of the base model.
      def chat(messages, max_new_tokens: 64, device: :auto, temperature: nil, top_p: nil, seed: nil)
        require_base_model

        Generation.run(
          model: base_model,
          messages: messages,
          adapter_path: path,
          max_new_tokens: max_new_tokens,
          device: device,
          temperature: temperature,
          top_p: top_p,
          seed: seed
        )
      end

      private

      # An adapter is a delta against specific base weights, so it cannot be applied without knowing which model it belongs to.
      def require_base_model
        return if base_model

        raise GenerationError, 'base model is unknown; pass base_model: to Adapter.load (or --model to the CLI)'
      end
    end
  end
end
