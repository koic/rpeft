# frozen_string_literal: true

require_relative 'grain/version'
require_relative 'grain/errors'
require_relative 'grain/arguments'
require_relative 'grain/device'
require_relative 'grain/directory'
require_relative 'grain/dataset'
require_relative 'grain/messages'
require_relative 'grain/generation'
require_relative 'grain/model'
require_relative 'grain/lora'
require_relative 'grain/lora/adapter'
require_relative 'grain/backend'
require_relative 'grain/diagnosis'

module Grain
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
