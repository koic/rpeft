# frozen_string_literal: true

require 'test_helper'

class TestDataset < Minitest::Test
  def setup
    @workdir = Dir.mktmpdir('grain_test')
    @file = File.join(@workdir, 'train.jsonl')
    File.write(@file, %({"text":"User: hi\\nAssistant: hello"}\n))
  end

  def teardown
    FileUtils.remove_entry(@workdir)
  end

  def test_coerce_wraps_a_path_string
    dataset = Grain::Dataset.coerce(@file)

    assert_instance_of Grain::Dataset, dataset
    assert_equal @file, dataset.path
  end

  def test_coerce_passes_a_dataset_through
    dataset = Grain::Dataset.new(@file)

    assert_same dataset, Grain::Dataset.coerce(dataset)
  end

  def test_coerce_builds_a_dataset_from_records
    dataset = Grain::Dataset.coerce([{text: 'first'}, {text: 'second'}])

    assert_equal :text, dataset.format

    records = File.readlines(dataset.path).map { |line| JSON.parse(line) }

    assert_equal(%w(first second), records.map { |record| record['text'] })
  end

  def test_from_records_accepts_messages_with_symbol_keys
    dataset = Grain::Dataset.from_records([{messages: [{role: 'user', content: 'hi'}]}])

    assert_equal :messages, dataset.format

    record = JSON.parse(File.readlines(dataset.path).first)

    assert_equal 'user', record.dig('messages', 0, 'role')
  end

  def test_from_records_renders_instruction_records
    dataset = Grain::Dataset.from_records([{instruction: 'Say hi', output: 'hi'}])

    assert_equal :instruction, dataset.format
    assert_includes JSON.parse(File.readlines(dataset.path).first)['text'], '### Instruction:'
  end

  def test_from_records_rejects_a_non_hash_record
    error = assert_raises(Grain::DatasetError) { Grain::Dataset.from_records(['just a string']) }

    assert_match(/must be a Hash/, error.message)
  end

  def test_from_records_rejects_an_empty_enumerable
    error = assert_raises(Grain::DatasetError) { Grain::Dataset.from_records([]) }

    assert_match(/no records/, error.message)
  end

  def test_rejects_a_missing_file
    error = assert_raises(Grain::DatasetError) {
      Grain::Dataset.new(File.join(@workdir, 'none.jsonl'))
    }

    assert_match(/not found/, error.message)
  end

  def test_rejects_an_empty_file
    File.write(@file, "\n")

    assert_raises(Grain::DatasetError) { Grain::Dataset.new(@file) }
  end

  def test_rejects_a_non_jsonl_file
    File.write(@file, "plain text\n")

    assert_raises(Grain::DatasetError) { Grain::Dataset.new(@file) }
  end

  def test_rejects_lines_without_a_text_key
    File.write(@file, %({"prompt":"hi"}\n))

    error = assert_raises(Grain::DatasetError) { Grain::Dataset.new(@file) }

    assert_match(/"text"/, error.message)
  end

  def test_detects_the_text_format
    assert_equal :text, Grain::Dataset.new(@file).format
  end

  def test_renders_an_instruction_dataset_into_text_format
    File.write(@file, <<~JSONL)
      {"instruction":"Say hi","output":"hi"}
      {"instruction":"Add numbers","input":"1 and 2","output":"3"}
    JSONL

    dataset = Grain::Dataset.new(@file)

    assert_equal :instruction, dataset.format
    refute_equal @file, dataset.path

    records = File.readlines(dataset.path).map { |line| JSON.parse(line) }

    assert_equal 2, records.size
    assert_equal "### Instruction:\nSay hi\n\n### Response:\nhi", records[0]['text']
    assert_includes records[1]['text'], "### Input:\n1 and 2"
  end

  def test_detects_the_messages_format_and_passes_the_file_through
    File.write(@file, %({"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}\n))

    dataset = Grain::Dataset.new(@file)

    assert_equal :messages, dataset.format
    assert_equal @file, dataset.path
    assert_equal 'messages', dataset.backend_format
  end

  def test_reports_text_as_the_backend_format_for_instruction_datasets
    File.write(@file, %({"instruction":"Say hi","output":"hi"}\n))

    assert_equal 'text', Grain::Dataset.new(@file).backend_format
  end

  def test_rejects_messages_that_are_not_an_array
    File.write(@file, %({"messages":"hi"}\n))

    error = assert_raises(Grain::DatasetError) { Grain::Dataset.new(@file) }

    assert_match(/"messages"/, error.message)
  end

  def test_rejects_messages_without_role_and_content
    File.write(@file, %({"messages":[{"speaker":"user","text":"hi"}]}\n))

    assert_raises(Grain::DatasetError) { Grain::Dataset.new(@file) }
  end

  def test_rejects_an_empty_messages_array
    File.write(@file, %({"messages":[]}\n))

    assert_raises(Grain::DatasetError) { Grain::Dataset.new(@file) }
  end

  def test_rejects_an_instruction_line_without_an_output
    File.write(@file, <<~JSONL)
      {"instruction":"Say hi","output":"hi"}
      {"instruction":"Broken"}
    JSONL

    error = assert_raises(Grain::DatasetError) { Grain::Dataset.new(@file) }

    assert_match(/line 2/, error.message)
  end
end
