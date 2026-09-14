# frozen_string_literal: true

module Grain
  module CLI
    # @api private
    class LoraTrain < Command
      PASSTHROUGH_OPTIONS = %i(rank alpha dropout target_modules epochs batch_size
                               learning_rate max_length device).freeze
      REQUIRED_OPTIONS = %i(model dataset output).freeze
      DEVICES = %w(auto cuda mps cpu).freeze

      private

      def option_parser(options)
        OptionParser.new do |o|
          o.banner = 'Usage: grain lora train --model MODEL --dataset FILE --output DIR [options]'
          o.on('--model MODEL', 'Base model (Hugging Face id or local path)') do |v|
            options[:model] = v
          end
          o.on('--dataset FILE', %(Training dataset (JSONL with a "text" field))) do |v|
            options[:dataset] = v
          end
          o.on('--output DIR', 'Directory to save the trained adapter') do |v|
            options[:output] = v
          end
          o.on('--rank N', Integer, 'LoRA rank r (default: 8)') do |v|
            options[:rank] = v
          end
          o.on('--alpha N', Float, 'LoRA alpha (default: 16)') do |v|
            options[:alpha] = v
          end
          o.on('--dropout F', Float, 'LoRA dropout (default: 0.05)') do |v|
            options[:dropout] = v
          end
          o.on('--target-modules LIST', Array,
               'Comma-separated module names (default: auto-detect)') do |v|
            options[:target_modules] = v
          end
          o.on('--epochs N', Integer, 'Training epochs (default: 3)') do |v|
            options[:epochs] = v
          end
          o.on('--batch-size N', Integer, 'Batch size (default: 4)') do |v|
            options[:batch_size] = v
          end
          o.on('--learning-rate F', Float, 'Learning rate (default: 0.0002)') do |v|
            options[:learning_rate] = v
          end
          o.on('--max-length N', Integer, 'Max token length (default: 512)') do |v|
            options[:max_length] = v
          end
          o.on('--device DEVICE', DEVICES,
               'auto, cuda, mps, or cpu (default: auto)') { |v| options[:device] = v.to_sym }
        end
      end

      def required_options
        REQUIRED_OPTIONS
      end

      def execute(options)
        train_options = {model: options[:model], dataset: options[:dataset],
                         output_dir: options[:output]}
        PASSTHROUGH_OPTIONS.each do |key|
          train_options[key] = options[key] if options.key?(key)
        end

        adapter = LoRA.train(**train_options)

        @out.puts "Adapter saved to #{adapter.path}"
        loss = adapter.metadata.dig('training', 'train_loss')
        @out.puts format('Final training loss: %.4f', loss) if loss
        0
      end
    end
  end
end
