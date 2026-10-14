# frozen_string_literal: true

module Grain
  module CLI
    # @api private
    class Doctor < Command
      private

      def option_parser(_options)
        OptionParser.new do |o|
          o.banner = 'Usage: grain doctor'
        end
      end

      def execute(_options)
        report = Grain.check
        render(report)
        report['status'] == 'ok' ? 0 : 1
      end

      def render(report)
        line 'Ruby', "#{report.dig('ruby', 'version')} (grain #{report.dig('ruby', 'grain')})"
        backend = report['backend'] || {}
        if backend['ok']
          render_python(report['python'])
          render_libraries(report['libraries'])
          render_device(report['device'])
        else
          line 'Backend', "NG #{backend['error']}"
        end

        # Uppercase for consistency with the OK/MISSING library markers;
        # the check API itself keeps the machine-readable lowercase value.
        line 'Status', report['status'].to_s.upcase
        render_hint(report)
      end

      def line(label, value)
        padded = "#{label}:".ljust(14)
        @out.puts "#{padded}#{value}"
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

      def render_hint(report)
        return unless report['status'] == 'degraded'

        @out.puts 'Hint: pip install --upgrade torch transformers peft datasets accelerate'
      end
    end
  end
end
