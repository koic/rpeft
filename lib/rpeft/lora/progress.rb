# frozen_string_literal: true

module RPEFT
  module LoRA
    # One step of a training run, as the block given to LoRA.train sees it. A training step carries loss and learning_rate,
    # an evaluation at the end of an epoch carries eval_loss, and whatever a step lacks is nil.
    class Progress
      attr_reader :step, :total_steps, :epoch, :loss, :eval_loss, :learning_rate

      def initialize(entry)
        @step = entry['step']
        @total_steps = entry['total_steps']
        @epoch = entry['epoch']
        @loss = entry['loss']
        @eval_loss = entry['eval_loss']
        @learning_rate = entry['learning_rate']
        freeze
      end
    end
  end
end
