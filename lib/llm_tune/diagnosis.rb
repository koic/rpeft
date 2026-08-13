# frozen_string_literal: true

module LLMTune
  # Builds the layered environment report behind LLMTune.check and `llm-tune doctor`.
  # Never raises: backend failures are folded into the report so the doctor can explain
  # a broken environment.
  class Diagnosis
    REQUIRED_LIBRARIES = %w(torch transformers peft datasets accelerate).freeze

    def initialize(backend: LLMTune.backend)
      @backend = backend
    end

    def report
      base = {'ruby' => {'version' => RUBY_VERSION, 'llm_tune' => VERSION}}
      begin
        diagnostics = @backend.diagnostics
      rescue Error => e
        return base.merge(
          'backend' => {'ok' => false, 'error' => e.message},
          'status' => 'unusable'
        )
      end

      base.merge(
        'backend' => {'ok' => true, 'version' => diagnostics['backend_version']},
        'python' => diagnostics['python'],
        'libraries' => diagnostics['libraries'],
        'device' => diagnostics['device'],
        'status' => status_of(diagnostics)
      )
    end

    private

    def status_of(diagnostics)
      libraries = diagnostics['libraries'] || {}
      missing = REQUIRED_LIBRARIES.reject { |name| libraries.dig(name, 'installed') }
      missing.empty? ? 'ok' : 'degraded'
    end
  end
end
