# frozen_string_literal: true

require 'test_helper'

class TestLoRATrain < LLMTuneTest
  def setup
    super
    @workdir = Dir.mktmpdir('llm_tune_test')
    @dataset = File.join(@workdir, 'train.jsonl')
    File.write(@dataset, %({"text":"User: hi\\nAssistant: hello"}\n))
  end

  def teardown
    FileUtils.remove_entry(@workdir)
    super
  end

  def test_returns_an_adapter_with_the_lora_configuration
    adapter = train(rank: 4, alpha: 8)

    assert_instance_of LLMTune::LoRA::Adapter, adapter
    assert_equal 'fake/model', adapter.base_model
    assert_equal 4, adapter.rank
    assert_equal 8, adapter.alpha
    assert_in_delta 0.05, adapter.dropout
    assert_equal File.join(@workdir, 'adapter'), adapter.path
  end

  def test_sends_the_training_configuration_to_the_backend
    train

    _name, config = @backend.calls.find { |call| call.first == :train_lora }

    assert_equal 'fake/model', config['base_model']
    assert_equal @dataset, config['dataset_path']
    assert_equal 8, config.dig('lora', 'r')
    assert_equal 16, config.dig('lora', 'alpha')
    assert_in_delta 0.05, config.dig('lora', 'dropout')
    assert_nil config.dig('lora', 'target_modules')
    assert_equal 3, config['epochs']
    assert_equal 4, config['batch_size']
    assert_in_delta 2e-4, config['learning_rate']
    assert_equal 512, config['max_length']
    assert_equal 'cpu', config['device']
  end

  def test_resolves_auto_device_before_calling_the_backend
    train

    assert_equal :detect_device, @backend.calls.first.first
    _name, config = @backend.calls.find { |call| call.first == :train_lora }
    assert_equal 'cpu', config['device']
  end

  def test_passes_an_explicit_device_through_without_detection
    train(device: :mps)

    refute(@backend.calls.any? { |call| call.first == :detect_device })
    _name, config = @backend.calls.find { |call| call.first == :train_lora }
    assert_equal 'mps', config['device']
  end

  def test_records_the_resolved_target_modules_from_the_backend
    adapter = train

    assert_equal %w(q_proj v_proj), adapter.target_modules
  end

  def test_writes_llm_tune_metadata_into_the_adapter_directory
    adapter = train(epochs: 2)

    metadata = JSON.parse(File.read(File.join(adapter.path, 'llm_tune.json')))

    assert_equal 1, metadata['schema_version']
    assert_equal 'lora_adapter', metadata['kind']
    assert_equal 'fake/model', metadata['base_model']
    assert_equal 8, metadata.dig('lora', 'rank')
    assert_equal %w(q_proj v_proj), metadata.dig('lora', 'target_modules')
    assert_equal 2, metadata.dig('training', 'epochs')
    assert_in_delta 2.5, metadata.dig('training', 'train_loss')
    assert_equal 'cpu', metadata.dig('training', 'device')
  end

  def test_creates_a_temporary_output_directory_by_default
    adapter = LLMTune::LoRA.train(model: 'fake/model', dataset: @dataset)

    assert File.directory?(adapter.path)
    assert_path_exists File.join(adapter.path, 'llm_tune.json')
  ensure
    FileUtils.remove_entry(adapter.path) if adapter
  end

  def test_rejects_a_non_positive_rank
    error = assert_raises(LLMTune::ConfigurationError) { train(rank: 0) }

    assert_match(/rank/, error.message)
  end

  def test_rejects_a_non_integer_rank
    assert_raises(LLMTune::ConfigurationError) { train(rank: '8') }
  end

  def test_rejects_an_out_of_range_dropout
    assert_raises(LLMTune::ConfigurationError) { train(dropout: 1.5) }
  end

  def test_rejects_a_non_array_target_modules
    assert_raises(LLMTune::ConfigurationError) { train(target_modules: 'q_proj') }
  end

  def test_rejects_a_non_positive_epochs
    assert_raises(LLMTune::ConfigurationError) { train(epochs: 0) }
  end

  def test_rejects_an_unknown_device
    error = assert_raises(LLMTune::ConfigurationError) { train(device: :tpu) }

    assert_match(/device/, error.message)
  end

  def test_raises_when_the_dataset_file_is_missing
    error = assert_raises(LLMTune::DatasetError) {
      LLMTune::LoRA.train(model: 'fake/model', dataset: File.join(@workdir, 'none.jsonl'))
    }

    assert_match(/not found/, error.message)
    assert_empty @backend.calls
  end

  def test_raises_when_the_dataset_is_empty
    File.write(@dataset, "\n")

    assert_raises(LLMTune::DatasetError) { train }
  end

  def test_raises_when_the_dataset_is_not_jsonl
    File.write(@dataset, "plain text\n")

    assert_raises(LLMTune::DatasetError) { train }
  end

  def test_disables_checkpoints_by_default
    train

    _name, config = @backend.calls.find { |call| call.first == :train_lora }

    assert_equal 'no', config['save_strategy']
    refute config['resume_from_checkpoint']
  end

  def test_saves_a_checkpoint_per_epoch_when_asked
    train(checkpoints: true)

    _name, config = @backend.calls.find { |call| call.first == :train_lora }

    assert_equal 'epoch', config['save_strategy']
  end

  def test_requests_a_resume_when_asked
    train(checkpoints: true, resume: true)

    _name, config = @backend.calls.find { |call| call.first == :train_lora }

    assert config['resume_from_checkpoint']
  end

  def test_rejects_resume_without_an_explicit_output_dir
    error = assert_raises(LLMTune::ConfigurationError) {
      LLMTune::LoRA.train(model: 'fake/model', dataset: @dataset, resume: true)
    }

    assert_match(/output_dir/, error.message)
    assert_empty @backend.calls
  end

  def test_sends_the_eval_dataset_path_when_given
    eval_file = File.join(@workdir, 'eval.jsonl')
    File.write(eval_file, %({"text":"User: bye\\nAssistant: goodbye"}\n))

    adapter = train(eval_dataset: eval_file)

    _name, config = @backend.calls.find { |call| call.first == :train_lora }

    assert_equal eval_file, config['eval_dataset_path']
    assert_in_delta 2.75, adapter.metadata.dig('training', 'eval_loss')
  end

  def test_omits_the_eval_dataset_path_by_default
    adapter = train

    _name, config = @backend.calls.find { |call| call.first == :train_lora }

    refute config.key?('eval_dataset_path')
    assert_nil adapter.metadata.dig('training', 'eval_loss')
  end

  def test_rejects_an_eval_dataset_in_a_different_format
    eval_file = File.join(@workdir, 'eval.jsonl')
    File.write(eval_file, %({"messages":[{"role":"user","content":"hi"}]}\n))

    error = assert_raises(LLMTune::ConfigurationError) { train(eval_dataset: eval_file) }

    assert_match(/eval_dataset format/, error.message)
  end

  def test_trains_from_an_array_of_records
    LLMTune::LoRA.train(
      model: 'fake/model',
      dataset: [{text: 'from memory'}],
      output_dir: File.join(@workdir, 'adapter')
    )

    _name, config = @backend.calls.find { |call| call.first == :train_lora }

    assert_equal 'from memory', JSON.parse(File.readlines(config['dataset_path']).first)['text']
  end

  def test_sends_the_text_dataset_format_by_default
    train

    _name, config = @backend.calls.find { |call| call.first == :train_lora }

    assert_equal 'text', config['dataset_format']
  end

  def test_sends_the_messages_dataset_format_and_the_original_file
    File.write(@dataset, %({"messages":[{"role":"user","content":"hi"}]}\n))

    train

    _name, config = @backend.calls.find { |call| call.first == :train_lora }

    assert_equal 'messages', config['dataset_format']
    assert_equal @dataset, config['dataset_path']
  end

  def test_trains_from_an_instruction_dataset
    File.write(@dataset, %({"instruction":"Say hi","output":"hi"}\n))

    train

    _name, config = @backend.calls.find { |call| call.first == :train_lora }
    first = JSON.parse(File.readlines(config['dataset_path']).first)

    assert_includes first['text'], '### Instruction:'
    assert_includes first['text'], '### Response:'
  end

  def test_raises_when_the_dataset_lines_lack_a_text_key
    File.write(@dataset, %({"prompt":"hi"}\n))

    error = assert_raises(LLMTune::DatasetError) { train }

    assert_match(/"text"/, error.message)
  end

  private

  def train(**options)
    LLMTune::LoRA.train(model: 'fake/model', dataset: @dataset,
                        output_dir: File.join(@workdir, 'adapter'), **options)
  end
end
