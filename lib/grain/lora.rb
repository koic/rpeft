# frozen_string_literal: true

require 'fileutils'
require 'json'
require 'time'
require 'tmpdir'

module Grain
  # LoRA fine-tuning. Keeps the base model weights W0 frozen and learns
  # a low-rank update delta W ~= B A of rank r:
  #
  #   W = W0 + B A
  #
  # The `rank:` option is exactly that r.
  module LoRA
    POSITIVE_INTEGER_OPTIONS = %i(rank epochs batch_size max_length).freeze
    POSITIVE_NUMBER_OPTIONS = %i(alpha learning_rate).freeze

    class << self
      def train(model:, dataset:, eval_dataset: nil, rank: 8, alpha: 16, dropout: 0.05, target_modules: nil, epochs: 3, batch_size: 4, learning_rate: 2e-4, max_length: 512, output_dir: nil, device: :auto, checkpoints: false, resume: false)
        options = {
          rank: rank,
          alpha: alpha,
          dropout: dropout,
          target_modules: target_modules,
          epochs: epochs,
          batch_size: batch_size,
          learning_rate: learning_rate,
          max_length: max_length,
          checkpoints: checkpoints,
          resume: resume
        }
        validate_options(options)
        validate_resume(resume, output_dir)
        training_dataset = Dataset.coerce(dataset)
        evaluation_dataset = coerce_eval_dataset(eval_dataset, training_dataset)
        adapter_dir = prepare_output_dir(output_dir)
        resolved_device = Device.resolve(device)

        options[:eval_dataset] = evaluation_dataset
        result = Grain.backend.train_lora(
          request_payload(model, training_dataset, adapter_dir, resolved_device, options)
        )

        build_adapter(
          adapter_dir: adapter_dir,
          model: model,
          dataset_name: training_dataset.name,
          device: resolved_device,
          options: options,
          result: result
        )
      rescue StandardError
        # A temporary directory is never returned to the caller, so a failed run leaves nothing anyone could
        # resume from. An explicit output_dir is kept for resume: true.
        FileUtils.rm_rf(adapter_dir) if adapter_dir && output_dir.nil?
        raise
      ensure
        # Datasets given as Dataset objects belong to the caller and may be reused for another run.
        training_dataset.discard if training_dataset && !training_dataset.equal?(dataset)
        evaluation_dataset.discard if evaluation_dataset && !evaluation_dataset.equal?(eval_dataset)
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

      # Without an explicit output_dir every run gets a fresh temporary directory, so there would never be a checkpoint of an earlier run to resume from.
      def validate_resume(resume, output_dir)
        return if !resume || !output_dir.nil?

        raise ConfigurationError, 'resume: true requires an explicit output_dir: to find the previous checkpoints'
      end

      # The backend tokenizes both datasets the same way, so mixing formats would silently mis-render the evaluation set.
      def coerce_eval_dataset(eval_dataset, training_dataset)
        return nil unless eval_dataset

        dataset = Dataset.coerce(eval_dataset)
        return dataset if dataset.backend_format == training_dataset.backend_format

        # Raised before train assigns the dataset, so its ensure clause cannot discard it.
        dataset.discard unless dataset.equal?(eval_dataset)
        raise ConfigurationError, "eval_dataset format (#{dataset.format}) must match dataset format (#{training_dataset.format})"
      end

      def prepare_output_dir(output_dir)
        return Dir.mktmpdir('grain-adapter-') unless output_dir

        # TRAINER_DIR marks a run that stopped before saving the adapter, which resume: true picks up.
        Directory.prepare(output_dir, markers: [*Adapter::REPLACE_MARKERS, Adapter::TRAINER_DIR])
      end

      def request_payload(model, dataset, adapter_dir, device, options)
        payload = {
          'base_model' => model,
          'dataset_path' => dataset.path,
          'dataset_format' => dataset.backend_format,
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
          'save_strategy' => options[:checkpoints] ? 'epoch' : 'no',
          'resume_from_checkpoint' => options[:resume],
          'lora' => {
            'r' => options[:rank],
            'alpha' => options[:alpha],
            'dropout' => options[:dropout],
            'target_modules' => options[:target_modules]
          }
        }
        payload['eval_dataset_path'] = options[:eval_dataset].path if options[:eval_dataset]

        payload
      end

      def build_adapter(adapter_dir:, model:, dataset_name:, device:, options:, result:)
        target_modules = result.dig('effective_config', 'lora', 'target_modules') || options[:target_modules]
        metadata = build_metadata(
          model: model,
          dataset_name: dataset_name,
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

      def build_metadata(model:, dataset_name:, device:, options:, target_modules:, result:)
        {
          'schema_version' => 1,
          'grain_version' => VERSION,
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
            'dataset' => dataset_name,
            'epochs' => options[:epochs],
            'batch_size' => options[:batch_size],
            'learning_rate' => options[:learning_rate],
            'max_length' => options[:max_length],
            'device' => result['device'] || device.to_s,
            'train_loss' => result['final_loss'],
            'eval_loss' => result['eval_loss'],
            'duration_seconds' => result['train_runtime_seconds']
          }
        }
      end
    end
  end
end
