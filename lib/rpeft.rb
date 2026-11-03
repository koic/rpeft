# frozen_string_literal: true

require_relative 'rpeft/version'
require_relative 'rpeft/errors'
require_relative 'rpeft/arguments'
require_relative 'rpeft/device'
require_relative 'rpeft/directory'
require_relative 'rpeft/dataset'
require_relative 'rpeft/messages'
require_relative 'rpeft/generation'
require_relative 'rpeft/model'
require_relative 'rpeft/lora'
require_relative 'rpeft/lora/adapter'
require_relative 'rpeft/backend'
require_relative 'rpeft/diagnosis'

module RPEFT
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
