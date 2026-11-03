# frozen_string_literal: true

module RPEFT
  # @api private
  #
  # Shape check for a chat transcript, shared by the dataset reader and the chat API.
  # Only the predicate is shared: each caller raises the error that fits its context.
  # String and Symbol keys are both accepted, since transcripts read from JSONL have
  # String keys while ones written in Ruby naturally have Symbol keys.
  module Messages
    # Chat templates write the role into the prompt as is, and some drop turns whose role they do not know,
    # so only the roles every template understands are accepted.
    ROLES = %w(system user assistant).freeze

    module_function

    def valid?(messages)
      return false if !messages.is_a?(Array) || messages.empty?

      messages.all? { |message|
        next false unless message.is_a?(Hash)

        role = message['role'] || message[:role]
        content = message['content'] || message[:content]

        ROLES.include?(role) && content.is_a?(String)
      }
    end
  end
end
