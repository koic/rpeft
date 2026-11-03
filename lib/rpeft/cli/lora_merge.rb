# frozen_string_literal: true

module RPEFT
  module CLI
    # @api private
    class LoRAMerge < Command
      REQUIRED_OPTIONS = %i(adapter output).freeze

      private

      def option_parser(options)
        OptionParser.new do |o|
          o.banner = 'Usage: rpeft lora merge --adapter DIR --output DIR [options]'
          o.on('--adapter DIR', 'Trained LoRA adapter directory') do |v|
            options[:adapter] = v
          end
          o.on('--output DIR', 'Directory to save the merged model') do |v|
            options[:output] = v
          end
          o.on('--model MODEL',
               'Base model override (defaults to the one recorded in the adapter)') do |v|
            options[:model] = v
          end
        end
      end

      def required_options
        REQUIRED_OPTIONS
      end

      def execute(options)
        adapter = LoRA::Adapter.load(options[:adapter], base_model: options[:model])
        model = adapter.merge(output_dir: options[:output])

        @out.puts "Merged model saved to #{model.path}"
        0
      end
    end
  end
end
