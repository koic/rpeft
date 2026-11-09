# frozen_string_literal: true

require_relative '../rpeft'
require_relative 'cli/command'
require_relative 'cli/lora_train'
require_relative 'cli/lora_merge'
require_relative 'cli/generate'
require_relative 'cli/doctor'
require_relative 'cli/export_gguf'

module RPEFT
  # Command line interface. Commands go through the public Ruby API only,
  # never straight to the backend.
  module CLI
    USAGE = <<~USAGE
      Usage: rpeft <command> [options]

      Commands:
        lora train   Fine-tune a base model with LoRA
        lora merge   Merge a LoRA adapter into its base model
        generate     Generate text from a base or fine-tuned model
        export gguf  Export a model to a GGUF file for llama.cpp and Ollama
        doctor       Diagnose the Python environment
        version      Print the RPEFT version
        help         Show this message

      Run `rpeft <command> --help` for command options.
    USAGE

    LORA_USAGE = <<~USAGE
      Usage: rpeft lora <subcommand> [options]

      Subcommands:
        train   Fine-tune a base model with LoRA
        merge   Merge a LoRA adapter into its base model
    USAGE

    EXPORT_USAGE = <<~USAGE
      Usage: rpeft export <format> [options]

      Formats:
        gguf    Export a model to a GGUF file for llama.cpp and Ollama
    USAGE

    class << self
      def run(argv, out: $stdout, err: $stderr)
        command, *rest = argv
        case command
        when 'lora' then run_lora(rest, out: out, err: err)
        when 'export' then run_export(rest, out: out, err: err)
        when 'generate' then Generate.run(rest, out: out, err: err)
        when 'doctor' then Doctor.run(rest, out: out, err: err)
        when 'version', '--version', '-v'
          out.puts VERSION
          0
        when nil, 'help', '--help', '-h'
          out.puts USAGE
          0
        else
          err.puts "rpeft: unknown command #{command.inspect}"
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
        when 'train' then LoRATrain.run(rest, out: out, err: err)
        when 'merge' then LoRAMerge.run(rest, out: out, err: err)
        when nil, 'help', '--help', '-h'
          out.puts LORA_USAGE
          0
        else
          err.puts "rpeft: unknown lora subcommand #{subcommand.inspect}"
          err.puts LORA_USAGE
          2
        end
      end

      def run_export(argv, out:, err:)
        format, *rest = argv

        case format
        when 'gguf' then ExportGGUF.run(rest, out: out, err: err)
        when nil, 'help', '--help', '-h'
          out.puts EXPORT_USAGE
          0
        else
          err.puts "rpeft: unknown export format #{format.inspect}"
          err.puts EXPORT_USAGE
          2
        end
      end
    end
  end
end
