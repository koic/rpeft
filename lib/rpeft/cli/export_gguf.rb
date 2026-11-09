# frozen_string_literal: true

module RPEFT
  module CLI
    # @api private
    class ExportGGUF < Command
      REQUIRED_OPTIONS = %i(model output).freeze

      private

      def option_parser(options)
        OptionParser.new do |o|
          o.banner = 'Usage: rpeft export gguf --model DIR --output FILE [options]'
          o.on('--model DIR', 'Model directory to export, such as a merged model') do |v|
            options[:model] = v
          end
          o.on('--output FILE', 'GGUF file to write') do |v|
            options[:output] = v
          end
          o.on('--quantize TYPE', "#{GGUF::TYPES.join(', ')}, in any case (default: q8_0)") do |v|
            type = v.downcase.to_sym
            raise OptionParser::InvalidArgument, v unless GGUF::TYPES.include?(type)

            options[:quantize] = type
          end
          o.on('--no-chat-template', 'Leave out the chat template, for a model fine-tuned on plain text') do
            options[:chat_template] = false
          end
          o.on('--llama-cpp DIR', 'llama.cpp checkout holding convert_hf_to_gguf.py (default: $LLAMA_CPP_DIR)') do |v|
            options[:llama_cpp] = v
          end
          o.on('--llama-quantize PATH', 'llama-quantize executable for q6_k, q5_k_m, and q4_k_m (default: from PATH)') do |v|
            options[:llama_quantize] = v
          end
        end
      end

      def required_options
        REQUIRED_OPTIONS
      end

      def execute(options)
        export_options = options.slice(:quantize, :chat_template, :llama_cpp, :llama_quantize)
        path = Model.new(path: options[:model]).export_gguf(options[:output], **export_options)

        @out.puts "GGUF saved to #{printable(path)}"
        0
      end
    end
  end
end
