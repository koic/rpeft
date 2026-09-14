# frozen_string_literal: true

require_relative 'backend/error_translation'

module Grain
  # Seam between the Ruby API and the Python backend.
  #
  # A backend implements four methods, each taking and returning plain Hashes with String keys:
  #
  #   train_lora(config)    -> Hash
  #   merge_lora(config)    -> Hash
  #   generate(config)      -> Hash
  #   detect_device(config) -> Hash
  #   diagnostics           -> Hash
  #
  # Backend::Python is the default implementation. Tests inject a fake via Grain.backend=.
  module Backend
    def self.default
      require_relative 'backend/python'

      Python.new
    end
  end
end
