# frozen_string_literal: true

require 'test_helper'

# Exports a merged tiny model to GGUF with a real llama.cpp checkout. Needs what the LoRA integration tests
# need, plus LLAMA_CPP_DIR pointing at a llama.cpp checkout and the converter's own dependencies, such as
# sentencepiece, installed in the same Python environment:
#
#   RPEFT_INTEGRATION=1 LLAMA_CPP_DIR=~/src/llama.cpp bundle exec rake test:integration
class TestGGUFIntegration < Minitest::Test
  # A Qwen2 model like the README's example, so the test takes the converter's Qwen2 path. The tiny Llama model
  # the LoRA tests use cannot be converted: its config sets pad_token_id to -1, which the converter rejects.
  TINY_MODEL = 'trl-internal-testing/tiny-Qwen2ForCausalLM-2.5'

  def setup
    skip('set RPEFT_INTEGRATION=1 to run integration tests') unless ENV['RPEFT_INTEGRATION'] == '1'
    skip('set LLAMA_CPP_DIR to a llama.cpp checkout to run the GGUF export test') if ENV.fetch('LLAMA_CPP_DIR', '').empty?

    RPEFT.backend = nil
    @workdir = Dir.mktmpdir('rpeft_integration')
  end

  def teardown
    RPEFT.backend = nil
    FileUtils.remove_entry(@workdir) if @workdir
  end

  def test_exports_a_merged_model_to_gguf_with_or_without_its_chat_template
    model = merged_model

    with_template = File.binread(model.export_gguf(File.join(@workdir, 'model.gguf')))
    without_template = File.binread(model.export_gguf(File.join(@workdir, 'model-plain.gguf'), chat_template: false))

    assert_equal 'GGUF', with_template[0, 4]
    assert_includes with_template, RPEFT::GGUF::CHAT_TEMPLATE_KEY
    assert_equal 'GGUF', without_template[0, 4]
    refute_includes without_template, RPEFT::GGUF::CHAT_TEMPLATE_KEY
  end

  private

  def merged_model
    dataset = File.join(@workdir, 'train.jsonl')
    File.write(dataset, [{text: 'Ruby is fun.'}, {text: 'Gems are libraries.'}].map(&:to_json).join("\n"))
    adapter = RPEFT::LoRA.train(
      model: TINY_MODEL, dataset: dataset, output_dir: File.join(@workdir, 'adapter'),
      rank: 2, epochs: 1, batch_size: 2, max_length: 32, device: :cpu
    )
    adapter.merge(output_dir: File.join(@workdir, 'merged'))
  end
end
