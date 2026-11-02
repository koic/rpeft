# frozen_string_literal: true

require 'optparse'

module Grain
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

        violation = usage_violation(options)
        return usage_error(violation, parser) if violation

        execute(options) || 0
      rescue Error => e
        @err.puts "grain: #{printable(e.message)}"
        1
      end

      private

      # Control characters other than newline and tab, made visible as \xNN. Model output and messages
      # carried over from Python are arbitrary text, and a terminal would act on an escape sequence in them
      # (moving the cursor, rewriting the screen or the clipboard) instead of showing it.
      def printable(text)
        text.to_s.scrub.gsub(/[\x00-\x08\x0B-\x1F\x7F\u0080-\u009F]/) { |char| format('\\x%02X', char.ord) }
      end

      def required_options
        []
      end

      # Commands override this for constraints that a required-option list cannot express;
      # a String return value is a usage error.
      def usage_violation(_options)
        nil
      end

      def usage_error(message, parser)
        @err.puts "grain: #{printable(message)}"
        @err.puts parser
        2
      end

      def format_options(keys)
        keys.map { |key| "--#{key.to_s.tr('_', '-')}" }.join(', ')
      end
    end
  end
end
