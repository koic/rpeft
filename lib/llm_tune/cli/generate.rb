# frozen_string_literal: true

module LLMTune
  module CLI
    # @api private
    class Generate < Command
      REQUIRED_OPTIONS = %i(model prompt).freeze
      PASSTHROUGH_OPTIONS = %i(max_new_tokens device).freeze
      DEVICES = %w(auto cuda mps cpu).freeze

      private

      def option_parser(options)
        OptionParser.new do |o|
          o.banner = 'Usage: llm-tune generate --model DIR_OR_ID --prompt TEXT [options]'
          o.on('--model MODEL', 'Model directory or Hugging Face model id') do |v|
            options[:model] = v
          end
          o.on('--prompt TEXT', 'Prompt to complete') do |v|
            options[:prompt] = v
          end
          o.on('--max-new-tokens N', Integer, 'Maximum number of tokens to generate (default: 64)') do |v|
            options[:max_new_tokens] = v
          end
          o.on('--device DEVICE', DEVICES, 'auto, cuda, mps, or cpu (default: auto)') { |v| options[:device] = v.to_sym }
        end
      end

      def required_options
        REQUIRED_OPTIONS
      end

      def execute(options)
        generate_options = {}
        PASSTHROUGH_OPTIONS.each do |key|
          generate_options[key] = options[key] if options.key?(key)
        end

        model = Model.new(path: options[:model])
        @out.puts model.generate(options[:prompt], **generate_options)
        0
      end
    end
  end
end
