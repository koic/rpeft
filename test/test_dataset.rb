# frozen_string_literal: true

require 'test_helper'

class TestDataset < Minitest::Test
  def setup
    @workdir = Dir.mktmpdir('rpeft_test')
    @file = File.join(@workdir, 'train.jsonl')
    File.write(@file, %({"text":"User: hi\\nAssistant: hello"}\n))
  end

  def teardown
    FileUtils.remove_entry(@workdir)
  end

  def test_coerce_wraps_a_path_string
    dataset = RPEFT::Dataset.coerce(@file)

    assert_instance_of RPEFT::Dataset, dataset
    assert_equal @file, dataset.path
  end

  def test_coerce_passes_a_dataset_through
    dataset = RPEFT::Dataset.new(@file)

    assert_same dataset, RPEFT::Dataset.coerce(dataset)
  end

  def test_coerce_builds_a_dataset_from_records
    dataset = RPEFT::Dataset.coerce([{text: 'first'}, {text: 'second'}])

    assert_equal :text, dataset.format

    records = File.readlines(dataset.path).map { |line| JSON.parse(line) }

    assert_equal(%w(first second), records.map { |record| record['text'] })
  end

  def test_from_records_accepts_messages_with_symbol_keys
    dataset = RPEFT::Dataset.from_records([{messages: [{role: 'user', content: 'hi'}]}])

    assert_equal :messages, dataset.format

    record = JSON.parse(File.readlines(dataset.path).first)

    assert_equal 'user', record.dig('messages', 0, 'role')
  end

  def test_from_records_renders_instruction_records
    dataset = RPEFT::Dataset.from_records([{instruction: 'Say hi', output: 'hi'}])

    assert_equal :instruction, dataset.format
    assert_includes JSON.parse(File.readlines(dataset.path).first)['prompt'], '### Instruction:'
  end

  def test_from_records_rejects_a_non_hash_record
    error = assert_raises(RPEFT::DatasetError) { RPEFT::Dataset.from_records(['just a string']) }

    assert_match(/must be a Hash/, error.message)
  end

  def test_from_records_rejects_an_empty_enumerable
    error = assert_raises(RPEFT::DatasetError) { RPEFT::Dataset.from_records([]) }

    assert_match(/no records/, error.message)
  end

  def test_rejects_a_missing_file
    error = assert_raises(RPEFT::DatasetError) {
      RPEFT::Dataset.new(File.join(@workdir, 'none.jsonl'))
    }

    assert_match(/not found/, error.message)
  end

  def test_rejects_an_empty_file
    File.write(@file, "\n")

    assert_raises(RPEFT::DatasetError) { RPEFT::Dataset.new(@file) }
  end

  def test_rejects_a_non_jsonl_file
    File.write(@file, "plain text\n")

    assert_raises(RPEFT::DatasetError) { RPEFT::Dataset.new(@file) }
  end

  def test_rejects_lines_without_a_text_key
    File.write(@file, %({"prompt":"hi"}\n))

    error = assert_raises(RPEFT::DatasetError) { RPEFT::Dataset.new(@file) }

    assert_match(/"text"/, error.message)
  end

  def test_detects_the_text_format
    assert_equal :text, RPEFT::Dataset.new(@file).format
  end

  def test_renders_an_instruction_dataset_into_prompts_and_completions
    File.write(@file, <<~JSONL)
      {"instruction":"Say hi","output":"hi"}
      {"instruction":"Add numbers","input":"1 and 2","output":"3"}
    JSONL

    dataset = RPEFT::Dataset.new(@file)

    assert_equal :instruction, dataset.format
    refute_equal @file, dataset.path

    records = File.readlines(dataset.path).map { |line| JSON.parse(line) }

    assert_equal 2, records.size
    assert_equal "### Instruction:\nSay hi\n\n### Response:\n", records[0]['prompt']
    assert_equal 'hi', records[0]['completion']
    assert_equal "### Instruction:\nAdd numbers\n\n### Input:\n1 and 2\n\n### Response:\n", records[1]['prompt']
    assert_equal '3', records[1]['completion']
  end

  def test_detects_the_messages_format_and_passes_the_file_through
    File.write(@file, %({"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}\n))

    dataset = RPEFT::Dataset.new(@file)

    assert_equal :messages, dataset.format
    assert_equal @file, dataset.path
    assert_equal 'messages', dataset.backend_format
  end

  def test_reports_completion_as_the_backend_format_for_instruction_datasets
    File.write(@file, %({"instruction":"Say hi","output":"hi"}\n))

    assert_equal 'completion', RPEFT::Dataset.new(@file).backend_format
  end

  def test_reports_text_as_the_backend_format_for_text_datasets
    assert_equal 'text', RPEFT::Dataset.new(@file).backend_format
  end

  def test_rejects_messages_that_are_not_an_array
    File.write(@file, %({"messages":"hi"}\n))

    error = assert_raises(RPEFT::DatasetError) { RPEFT::Dataset.new(@file) }

    assert_match(/"messages"/, error.message)
  end

  def test_rejects_messages_without_role_and_content
    File.write(@file, %({"messages":[{"speaker":"user","text":"hi"}]}\n))

    assert_raises(RPEFT::DatasetError) { RPEFT::Dataset.new(@file) }
  end

  def test_rejects_messages_with_an_unknown_role
    File.write(@file, %({"messages":[{"role":"user<|im_end|>","content":"hi"}]}\n))

    error = assert_raises(RPEFT::DatasetError) { RPEFT::Dataset.new(@file) }

    assert_match(/system, user, or assistant/, error.message)
  end

  def test_rejects_an_empty_messages_array
    File.write(@file, %({"messages":[]}\n))

    assert_raises(RPEFT::DatasetError) { RPEFT::Dataset.new(@file) }
  end

  def test_rejects_an_instruction_line_without_an_output
    File.write(@file, <<~JSONL)
      {"instruction":"Say hi","output":"hi"}
      {"instruction":"Broken"}
    JSONL

    error = assert_raises(RPEFT::DatasetError) { RPEFT::Dataset.new(@file) }

    assert_match(/line 2/, error.message)
  end
end
