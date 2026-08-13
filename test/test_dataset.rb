# frozen_string_literal: true

require 'test_helper'

class TestDataset < Minitest::Test
  def setup
    @workdir = Dir.mktmpdir('llm_tune_test')
    @file = File.join(@workdir, 'train.jsonl')
    File.write(@file, %({"text":"User: hi\\nAssistant: hello"}\n))
  end

  def teardown
    FileUtils.remove_entry(@workdir)
  end

  def test_coerce_wraps_a_path_string
    dataset = LLMTune::Dataset.coerce(@file)

    assert_instance_of LLMTune::Dataset, dataset
    assert_equal @file, dataset.path
  end

  def test_coerce_passes_a_dataset_through
    dataset = LLMTune::Dataset.new(@file)

    assert_same dataset, LLMTune::Dataset.coerce(dataset)
  end

  def test_rejects_a_missing_file
    error = assert_raises(LLMTune::DatasetError) {
      LLMTune::Dataset.new(File.join(@workdir, 'none.jsonl'))
    }

    assert_match(/not found/, error.message)
  end

  def test_rejects_an_empty_file
    File.write(@file, "\n")

    assert_raises(LLMTune::DatasetError) { LLMTune::Dataset.new(@file) }
  end

  def test_rejects_a_non_jsonl_file
    File.write(@file, "plain text\n")

    assert_raises(LLMTune::DatasetError) { LLMTune::Dataset.new(@file) }
  end

  def test_rejects_lines_without_a_text_key
    File.write(@file, %({"prompt":"hi"}\n))

    error = assert_raises(LLMTune::DatasetError) { LLMTune::Dataset.new(@file) }

    assert_match(/"text"/, error.message)
  end
end
