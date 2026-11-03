# frozen_string_literal: true

require 'pathname'

module RPEFT
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

    # A base model: a Hub id is kept verbatim, while a local directory is made absolute. The model is
    # recorded in the adapter and loaded again later, possibly from another directory, where a relative
    # path would name something else.
    def model(value, name)
      model = path(value, name)
      File.directory?(model) ? File.expand_path(model) : model
    end
  end
end
