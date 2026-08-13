# frozen_string_literal: true

require_relative 'llm_tune/version'
require_relative 'llm_tune/errors'
require_relative 'llm_tune/device'
require_relative 'llm_tune/directory'
require_relative 'llm_tune/model'
require_relative 'llm_tune/lora'
require_relative 'llm_tune/lora/adapter'
require_relative 'llm_tune/backend'
require_relative 'llm_tune/diagnosis'

module LLMTune
  DEVICES = %i(cuda mps cpu).freeze

  class << self
    # The backend seam. Assign a fake in tests; assign nil to restore the default PyCall-based backend.
    attr_writer :backend

    def backend
      @backend ||= Backend.default
    end

    # The compute device that training will use, preferring CUDA over MPS over CPU.
    def device
      backend.detect_device({})['device'].to_sym
    end

    def check
      Diagnosis.new(backend: backend).report
    end
  end
end
