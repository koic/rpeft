# frozen_string_literal: true

require 'test_helper'

# Runs a real LoRA fine-tune with a tiny model on CPU. Requires
# a Python environment with torch/transformers/peft/datasets/accelerate
# and network access for the first model download (a few MB, cached
# afterwards):
#
#   GRAIN_INTEGRATION=1 bundle exec rake test:integration
#
# The model has random weights; this test asserts the mechanics of
# the train and merge lifecycle, never output quality. The Llama
# architecture is chosen so that target_modules: nil exercises PEFT's
# per-architecture auto-detection, the same path real users hit.
class TestLoRAIntegration < Minitest::Test
  TINY_MODEL = 'hf-internal-testing/tiny-random-LlamaForCausalLM'

  def setup
    skip('set GRAIN_INTEGRATION=1 to run integration tests') unless ENV['GRAIN_INTEGRATION'] == '1'

    Grain.backend = nil
    @workdir = Dir.mktmpdir('grain_integration')
    @dataset = File.join(@workdir, 'train.jsonl')
    File.write(@dataset, [
      {text: "Ruby is a programmer's best friend."},
      {text: 'Matz designed Ruby for developer happiness.'},
      {text: 'RuboCop keeps Ruby code consistent.'},
      {text: 'Gems are packaged Ruby libraries.'}
    ].map(&:to_json).join("\n"))
  end

  def teardown
    Grain.backend = nil
    FileUtils.remove_entry(@workdir) if @workdir
  end

  def test_lora_train_and_merge_produce_loadable_directories
    adapter = Grain::LoRA.train(
      model: TINY_MODEL,
      dataset: @dataset,
      output_dir: File.join(@workdir, 'adapter'),
      rank: 2,
      epochs: 1,
      batch_size: 2,
      max_length: 64,
      device: :cpu
    )

    assert_path_exists File.join(adapter.path, 'adapter_config.json')
    assert_path_exists File.join(adapter.path, 'adapter_model.safetensors')
    assert_path_exists File.join(adapter.path, 'grain.json')
    assert_kind_of Numeric, adapter.metadata.dig('training', 'train_loss')
    assert_equal 'cpu', adapter.metadata.dig('training', 'device')
    refute_empty adapter.target_modules

    adapter_completion = adapter.generate('Ruby is', max_new_tokens: 8, device: :cpu)

    assert_instance_of String, adapter_completion

    model = adapter.merge(output_dir: File.join(@workdir, 'merged'))

    assert_path_exists File.join(model.path, 'config.json')
    assert_path_exists File.join(model.path, 'model.safetensors')
    assert_path_exists File.join(model.path, 'tokenizer_config.json')
    refute_empty JSON.parse(File.read(File.join(model.path, 'config.json')))['model_type']

    completion = model.generate('Ruby is', max_new_tokens: 8, device: :cpu)

    assert_instance_of String, completion
  end

  def test_lora_train_reports_eval_loss_when_an_eval_dataset_is_given
    adapter = Grain::LoRA.train(
      model: TINY_MODEL,
      dataset: @dataset,
      eval_dataset: @dataset,
      output_dir: File.join(@workdir, 'eval-adapter'),
      rank: 2,
      epochs: 1,
      batch_size: 2,
      max_length: 64,
      device: :cpu
    )

    assert_kind_of Numeric, adapter.metadata.dig('training', 'eval_loss')
  end

  def test_lora_train_accepts_a_messages_dataset
    dataset = File.join(@workdir, 'chat.jsonl')
    File.write(dataset, [
      {messages: [{role: 'user', content: 'What is Ruby?'}, {role: 'assistant', content: 'A programming language.'}]},
      {messages: [{role: 'user', content: 'Who made Ruby?'}, {role: 'assistant', content: 'Matz.'}]}
    ].map(&:to_json).join("\n"))

    adapter = Grain::LoRA.train(
      model: TINY_MODEL,
      dataset: dataset,
      output_dir: File.join(@workdir, 'chat-adapter'),
      rank: 2,
      epochs: 1,
      batch_size: 2,
      max_length: 64,
      device: :cpu
    )

    assert_path_exists File.join(adapter.path, 'adapter_model.safetensors')
    assert_kind_of Numeric, adapter.metadata.dig('training', 'train_loss')

    # The tiny model has no chat template, so this covers the fallback rendering.
    reply = adapter.chat([{role: 'user', content: 'What is Ruby?'}], max_new_tokens: 8, device: :cpu)

    assert_instance_of String, reply
  end
end
