# frozen_string_literal: true

require 'json'
require 'time'
require 'tmpdir'

module LLMTune
  # LoRA fine-tuning. Keeps the base model weights W frozen and learns
  # a low-rank update delta W ~= B A of rank r:
  #
  #   W' = W + B A
  #
  # The `rank:` option is exactly that r.
  module LoRA
    POSITIVE_INTEGER_OPTIONS = %i(rank epochs batch_size max_length).freeze
    POSITIVE_NUMBER_OPTIONS = %i(alpha learning_rate).freeze

    class << self
      def train(model:, dataset:, rank: 8, alpha: 16, dropout: 0.05, target_modules: nil, epochs: 3, batch_size: 4, learning_rate: 2e-4, max_length: 512, output_dir: nil, device: :auto)
        options = {
          rank: rank,
          alpha: alpha,
          dropout: dropout,
          target_modules: target_modules,
          epochs: epochs,
          batch_size: batch_size,
          learning_rate: learning_rate,
          max_length: max_length
        }
        validate_options(options)
        dataset_path = Dataset.coerce(dataset).path
        resolved_device = Device.resolve(device)
        adapter_dir = prepare_output_dir(output_dir)

        result = LLMTune.backend.train_lora(
          request_payload(model, dataset_path, adapter_dir, resolved_device, options)
        )

        build_adapter(
          adapter_dir: adapter_dir,
          model: model,
          dataset_path: dataset_path,
          device: resolved_device,
          options: options,
          result: result
        )
      end

      private

      def validate_options(options)
        POSITIVE_INTEGER_OPTIONS.each do |key|
          value = options[key]
          if !value.is_a?(Integer) || !value.positive?
            raise ConfigurationError, "#{key} must be a positive Integer, got #{value.inspect}"
          end
        end
        POSITIVE_NUMBER_OPTIONS.each do |key|
          value = options[key]
          if !value.is_a?(Numeric) || !value.positive?
            raise ConfigurationError, "#{key} must be a positive number, got #{value.inspect}"
          end
        end
        validate_dropout(options[:dropout])
        validate_target_modules(options[:target_modules])
      end

      def validate_dropout(dropout)
        return if dropout.is_a?(Numeric) && dropout >= 0 && dropout < 1

        raise ConfigurationError, "dropout must be a number within 0.0...1.0, got #{dropout.inspect}"
      end

      def validate_target_modules(target_modules)
        return if target_modules.nil?
        return if target_modules.is_a?(Array) && !target_modules.empty? && target_modules.all?(String)

        raise ConfigurationError, "target_modules must be nil (auto) or a non-empty Array of Strings, got #{target_modules.inspect}"
      end

      def prepare_output_dir(output_dir)
        dir = File.expand_path(output_dir || Dir.mktmpdir('llm_tune-adapter-'))
        FileUtils.mkdir_p(dir)

        dir
      end

      def request_payload(model, dataset_path, adapter_dir, device, options)
        {
          'base_model' => model,
          'dataset_path' => dataset_path,
          'output_dir' => adapter_dir,
          'text_field' => 'text',
          'device' => device.to_s,
          'max_length' => options[:max_length],
          'epochs' => options[:epochs],
          'batch_size' => options[:batch_size],
          'learning_rate' => options[:learning_rate],
          'logging_steps' => 5,
          'seed' => 42,
          'trust_remote_code' => false,
          'lora' => {
            'r' => options[:rank],
            'alpha' => options[:alpha],
            'dropout' => options[:dropout],
            'target_modules' => options[:target_modules]
          }
        }
      end

      def build_adapter(adapter_dir:, model:, dataset_path:, device:, options:, result:)
        target_modules = result.dig('effective_config', 'lora', 'target_modules') || options[:target_modules]
        metadata = build_metadata(
          model: model,
          dataset_path: dataset_path,
          device: device,
          options: options,
          target_modules: target_modules,
          result: result
        )
        File.write(File.join(adapter_dir, Adapter::METADATA_FILE), "#{JSON.pretty_generate(metadata)}\n")

        Adapter.new(
          path: adapter_dir,
          base_model: model,
          rank: options[:rank],
          alpha: options[:alpha],
          dropout: options[:dropout],
          target_modules: target_modules,
          metadata: metadata
        )
      end

      def build_metadata(model:, dataset_path:, device:, options:, target_modules:, result:)
        {
          'schema_version' => 1,
          'llm_tune_version' => VERSION,
          'kind' => 'lora_adapter',
          'base_model' => model,
          'created_at' => Time.now.iso8601,
          'lora' => {
            'rank' => options[:rank],
            'alpha' => options[:alpha],
            'dropout' => options[:dropout],
            'target_modules' => target_modules
          },
          'training' => {
            'dataset' => dataset_path,
            'epochs' => options[:epochs],
            'batch_size' => options[:batch_size],
            'learning_rate' => options[:learning_rate],
            'max_length' => options[:max_length],
            'device' => result['device'] || device.to_s,
            'train_loss' => result['final_loss'],
            'duration_seconds' => result['train_runtime_seconds']
          }
        }
      end
    end
  end
end
