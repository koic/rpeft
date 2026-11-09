# frozen_string_literal: true

module RPEFT
  module CLI
    # @api private
    class Doctor < Command
      private

      def option_parser(_options)
        OptionParser.new do |o|
          o.banner = 'Usage: rpeft doctor'
        end
      end

      def execute(_options)
        report = RPEFT.check
        render(report)
        report['status'] == 'ok' ? 0 : 1
      end

      def render(report)
        line 'Ruby', "#{report.dig('ruby', 'version')} (rpeft #{report.dig('ruby', 'rpeft')})"
        backend = report['backend'] || {}
        if backend['ok']
          render_python(report['python'])
          render_libraries(report['libraries'])
          render_device(report['device'])
        else
          line 'Backend', "NG #{backend['error']}"
        end
        render_gguf(report['gguf'])

        # Uppercase for consistency with the OK/MISSING library markers;
        # the check API itself keeps the machine-readable lowercase value.
        line 'Status', report['status'].to_s.upcase
        render_hint(report)
      end

      # Values come from the environment, such as LLAMA_CPP_DIR and the directories on PATH, which may hold
      # escape sequences a terminal would act on.
      def line(label, value)
        padded = "#{label}:".ljust(14)
        @out.puts "#{padded}#{printable(value)}"
      end

      def render_python(python)
        python ||= {}
        line 'Python', "#{python['version']} (#{python['executable']})"
      end

      def render_libraries(libraries)
        (libraries || {}).each do |name, info|
          if !info['installed']
            line name, 'MISSING'
          elsif info['outdated']
            line name, "OUTDATED #{info['version']} (needs #{info['minimum']} or later)"
          else
            line name, "OK #{info['version']}"
          end
        end
      end

      def render_device(device)
        return line('Device', 'unknown (torch is not installed)') unless device

        cuda = device.dig('cuda', 'available') ? 'yes' : 'no'
        mps = device.dig('mps', 'available') ? 'yes' : 'no'
        line 'Device', "#{device['recommended']} (CUDA: #{cuda}, MPS: #{mps})"
      end

      def render_gguf(gguf)
        gguf ||= {}

        if gguf['converter']
          line 'llama.cpp', "OK #{gguf['llama_cpp_dir']}"
        elsif gguf['llama_cpp_dir']
          line 'llama.cpp', "NG no #{GGUF::CONVERTER} in #{gguf['llama_cpp_dir']} (LLAMA_CPP_DIR)"
        else
          line 'llama.cpp', 'not configured (optional, set LLAMA_CPP_DIR for GGUF export)'
        end

        line 'Quantizer', gguf['quantizer'] ? "OK #{gguf['quantizer']}" : "no #{GGUF::QUANTIZER} (optional, for q6_k, q5_k_m, and q4_k_m)"
      end

      def render_hint(report)
        return unless report['status'] == 'degraded'

        @out.puts 'Hint: pip install --upgrade torch transformers peft datasets accelerate'
      end
    end
  end
end
