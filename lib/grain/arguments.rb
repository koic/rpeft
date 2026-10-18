# frozen_string_literal: true

require 'pathname'

module Grain
  # @api private
  #
  # Argument checks shared by the public API, so that a wrong argument raises ConfigurationError
  # in Ruby instead of an arbitrary error from deep inside Ruby or the backend.
  module Arguments
    module_function

    # A local path or a Hugging Face Hub id, given as a String or a Pathname.
    def path(value, name)
      path = value.is_a?(Pathname) ? value.to_s : value
      return path if path.is_a?(String) && !path.empty?

      raise ConfigurationError, "#{name} must be a non-empty String or Pathname, got #{value.inspect}"
    end
  end
end
