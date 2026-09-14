# frozen_string_literal: true

module Grain
  # @api private
  module Device
    def self.resolve(device)
      device = device.to_sym
      return Grain.device if device == :auto

      unless DEVICES.include?(device)
        raise ConfigurationError, "unknown device: #{device.inspect} (expected :auto, :cuda, :mps, or :cpu)"
      end

      device
    end
  end
end
