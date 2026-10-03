# frozen_string_literal: true

# LoRA fine-tuning from Ruby, end to end:
#
#   base model -> (train) -> LoRA adapter -> (merge) -> merged model
#
# and a before/after comparison of the same prompt. Run from
# the repository root with a prepared Python environment:
#
#   bundle exec ruby examples/train_and_merge.rb

require 'grain'

BASE_MODEL = 'Qwen/Qwen2.5-Coder-0.5B'
PROMPT = "User: How do I reverse an array?\nAssistant:"

puts "Device: #{Grain.device}"

puts
puts '--- Before (base model) ---'
base = Grain::Model.new(path: BASE_MODEL)
puts base.generate(PROMPT, max_new_tokens: 60)

adapter = Grain::LoRA.train(
  model: BASE_MODEL,
  dataset: File.expand_path('ruby_answers.jsonl', __dir__),
  rank: 8, # The LoRA rank r in W = W0 + B A.
  alpha: 16,
  dropout: 0.05,
  epochs: 10 # The 40-sample dataset trains in seconds, so re-read it often.
)
adapter.save(File.expand_path('ruby-lora', __dir__))
puts "Adapter: #{adapter.path} (rank #{adapter.rank})"

model = adapter.merge
model.save(File.expand_path('ruby-model', __dir__))
puts "Merged model: #{model.path}"

puts
puts '--- After (fine-tuned model) ---'
puts model.generate(PROMPT, max_new_tokens: 60)
