# frozen_string_literal: true

module LLMTune
  # @api private
  #
  # Shape check for a chat transcript, shared by the dataset reader and the chat API.
  # Only the predicate is shared: each caller raises the error that fits its context.
  # String and Symbol keys are both accepted, since transcripts read from JSONL have
  # String keys while ones written in Ruby naturally have Symbol keys.
  module Messages
    module_function

    def valid?(messages)
      return false if !messages.is_a?(Array) || messages.empty?

      messages.all? { |message|
        next false unless message.is_a?(Hash)

        role = message['role'] || message[:role]
        content = message['content'] || message[:content]

        role.is_a?(String) && content.is_a?(String)
      }
    end
  end
end
