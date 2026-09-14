# frozen_string_literal: true

require_relative '../grain'
require_relative 'cli/command'
require_relative 'cli/lora_train'
require_relative 'cli/lora_merge'
require_relative 'cli/generate'
require_relative 'cli/doctor'

module Grain
  # Command line interface. Commands go through the public Ruby API only,
  # never straight to the backend.
  module CLI
    USAGE = <<~USAGE
      Usage: grain <command> [options]

      Commands:
        lora train   Fine-tune a base model with LoRA
        lora merge   Merge a LoRA adapter into its base model
        generate     Generate text from a base or fine-tuned model
        doctor       Diagnose the Python environment
        version      Print the grain version
        help         Show this message

      Run `grain <command> --help` for command options.
    USAGE

    LORA_USAGE = <<~USAGE
      Usage: grain lora <subcommand> [options]

      Subcommands:
        train   Fine-tune a base model with LoRA
        merge   Merge a LoRA adapter into its base model
    USAGE

    class << self
      def run(argv, out: $stdout, err: $stderr)
        command, *rest = argv
        case command
        when 'lora' then run_lora(rest, out: out, err: err)
        when 'generate' then Generate.run(rest, out: out, err: err)
        when 'doctor' then Doctor.run(rest, out: out, err: err)
        when 'version', '--version', '-v'
          out.puts VERSION
          0
        when nil, 'help', '--help', '-h'
          out.puts USAGE
          0
        else
          err.puts "grain: unknown command #{command.inspect}"
          err.puts USAGE
          2
        end
      rescue Interrupt
        err.puts 'Interrupted'
        130
      end

      private

      def run_lora(argv, out:, err:)
        subcommand, *rest = argv
        case subcommand
        when 'train' then LoraTrain.run(rest, out: out, err: err)
        when 'merge' then LoraMerge.run(rest, out: out, err: err)
        when nil, 'help', '--help', '-h'
          out.puts LORA_USAGE
          0
        else
          err.puts "grain: unknown lora subcommand #{subcommand.inspect}"
          err.puts LORA_USAGE
          2
        end
      end
    end
  end
end
