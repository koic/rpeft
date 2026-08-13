# frozen_string_literal: true

require 'optparse'

module LLMTune
  module CLI
    # @api private
    class Command
      def self.run(argv, out: $stdout, err: $stderr)
        new(out: out, err: err).run(argv)
      end

      def initialize(out:, err:)
        @out = out
        @err = err
      end

      def run(argv)
        options = {}
        parser = option_parser(options)
        help_requested = false
        parser.on_tail('-h', '--help', 'Show this help') do
          help_requested = true
        end

        begin
          parser.parse(argv)
        rescue OptionParser::ParseError => e
          return usage_error(e.message, parser)
        end
        if help_requested
          @out.puts parser
          return 0
        end

        missing = required_options.reject { |key| options.key?(key) }
        return usage_error("missing required option(s): #{format_options(missing)}", parser) unless missing.empty?

        execute(options) || 0
      rescue Error => e
        @err.puts "llm-tune: #{e.message}"
        1
      end

      private

      def required_options
        []
      end

      def usage_error(message, parser)
        @err.puts "llm-tune: #{message}"
        @err.puts parser
        2
      end

      def format_options(keys)
        keys.map { |key| "--#{key.to_s.tr('_', '-')}" }.join(', ')
      end
    end
  end
end
