# frozen_string_literal: true

require 'test_helper'

class TestLoRAAdapter < RPEFTTest
  def setup
    super
    @workdir = Dir.mktmpdir('rpeft_test')
    @adapter_dir = File.join(@workdir, 'adapter')
    FileUtils.mkdir_p(@adapter_dir)
  end

  def teardown
    FileUtils.remove_entry(@workdir)
    super
  end

  def test_load_reads_rpeft_metadata
    write_metadata

    adapter = RPEFT::LoRA::Adapter.load(@adapter_dir)

    assert_equal 'meta/base', adapter.base_model
    assert_equal 8, adapter.rank
    assert_equal 16, adapter.alpha
    assert_in_delta 0.05, adapter.dropout
    assert_equal %w(q_proj v_proj), adapter.target_modules
    assert_in_delta 1.23, adapter.metadata.dig('training', 'train_loss')
  end

  def test_load_prefers_an_explicit_base_model
    write_metadata

    adapter = RPEFT::LoRA::Adapter.load(@adapter_dir, base_model: 'override/base')

    assert_equal 'override/base', adapter.base_model
  end

  def test_load_falls_back_to_a_peft_adapter_config
    peft_config = {
      'base_model_name_or_path' => 'peft/base',
      'r' => 4, 'lora_alpha' => 8, 'lora_dropout' => 0.1,
      'target_modules' => %w(q_proj v_proj)
    }
    File.write(File.join(@adapter_dir, 'adapter_config.json'), JSON.generate(peft_config))

    adapter = RPEFT::LoRA::Adapter.load(@adapter_dir)

    assert_equal 'peft/base', adapter.base_model
    assert_equal 4, adapter.rank
    assert_equal 8, adapter.alpha
    assert_in_delta 0.1, adapter.dropout
    assert_empty adapter.metadata
  end

  def test_load_raises_for_a_directory_without_adapter_files
    error = assert_raises(RPEFT::ModelLoadError) { RPEFT::LoRA::Adapter.load(@adapter_dir) }

    assert_match(/not a LoRA adapter directory/, error.message)
  end

  def test_save_moves_the_adapter_directory
    File.write(File.join(@adapter_dir, 'dummy'), 'x')
    adapter = new_adapter
    dest = File.join(@workdir, 'moved')

    assert_same adapter, adapter.save(dest)
    assert_equal dest, adapter.path
    assert_path_exists File.join(dest, 'dummy')
    refute File.exist?(@adapter_dir)
  end

  def test_save_to_its_own_path_keeps_the_adapter
    File.write(File.join(@adapter_dir, 'rpeft_metadata.json'), '{}')
    adapter = new_adapter

    adapter.save(@adapter_dir)

    assert_equal @adapter_dir, adapter.path
    assert_path_exists File.join(@adapter_dir, 'rpeft_metadata.json')
  end

  def test_save_to_its_own_path_through_a_symlink_keeps_the_adapter
    File.write(File.join(@adapter_dir, 'rpeft_metadata.json'), '{}')
    link = File.join(@workdir, 'link')
    File.symlink(@adapter_dir, link)

    new_adapter.save(link)

    assert_path_exists File.join(@adapter_dir, 'rpeft_metadata.json')
  end

  def test_save_refuses_a_destination_that_contains_the_adapter
    File.write(File.join(@workdir, 'rpeft_metadata.json'), '{}')
    File.write(File.join(@adapter_dir, 'dummy'), 'x')

    error = assert_raises(RPEFT::ConfigurationError) { new_adapter.save(@workdir) }

    assert_match(/contains/, error.message)
    assert_path_exists File.join(@adapter_dir, 'dummy')
  end

  def test_save_refuses_a_destination_inside_the_adapter
    File.write(File.join(@adapter_dir, 'dummy'), 'x')

    assert_raises(RPEFT::ConfigurationError) do
      new_adapter.save(File.join(@adapter_dir, 'nested'))
    end
    assert_path_exists File.join(@adapter_dir, 'dummy')
  end

  def test_save_refuses_to_overwrite_an_unrelated_existing_directory
    dest = File.join(@workdir, 'existing')
    FileUtils.mkdir_p(dest)

    error = assert_raises(RPEFT::ConfigurationError) { new_adapter.save(dest) }

    assert_match(/already exists/, error.message)
  end

  def test_save_replaces_a_previously_saved_adapter
    File.write(File.join(@adapter_dir, 'dummy'), 'new')
    dest = File.join(@workdir, 'saved')
    FileUtils.mkdir_p(dest)
    File.write(File.join(dest, 'rpeft_metadata.json'), '{}')
    File.write(File.join(dest, 'stale'), 'old')

    new_adapter.save(dest)

    assert_path_exists File.join(dest, 'dummy')
    refute File.exist?(File.join(dest, 'stale'))
  end

  def test_merge_calls_the_backend_and_returns_a_model
    merged_dir = File.join(@workdir, 'merged')

    model = new_adapter.merge(output_dir: merged_dir)

    assert_instance_of RPEFT::Model, model
    assert_equal merged_dir, model.path
    assert_equal 'fake/model', model.base_model

    name, config = @backend.calls.last
    assert_equal :merge_lora, name
    assert_equal @adapter_dir, config['adapter_path']
    assert_equal merged_dir, config['output_dir']
    assert_equal 'fake/model', config['base_model']
  end

  def test_merge_marks_the_merged_model_so_a_later_save_can_replace_it
    merged_dir = File.join(@workdir, 'merged')

    new_adapter.merge(output_dir: merged_dir)

    metadata = JSON.parse(File.read(File.join(merged_dir, 'rpeft_metadata.json')))
    assert_equal 'merged_model', metadata['kind']
    assert_equal 'fake/model', metadata['base_model']
  end

  def test_merge_refuses_an_output_dir_holding_unrelated_files
    output = File.join(@workdir, 'app')
    FileUtils.mkdir_p(output)
    File.write(File.join(output, 'config.json'), %({"app": "settings"}))

    error = assert_raises(RPEFT::ConfigurationError) { new_adapter.merge(output_dir: output) }

    assert_match(/not empty/, error.message)
    assert_equal %({"app": "settings"}), File.read(File.join(output, 'config.json'))
    refute File.exist?(File.join(output, 'rpeft_metadata.json'))
    assert_empty @backend.calls
  end

  def test_merge_writes_again_into_a_model_it_merged_before
    output = File.join(@workdir, 'merged')
    new_adapter.merge(output_dir: output)

    assert_equal output, new_adapter.merge(output_dir: output).path
  end

  def test_merge_uses_a_temporary_directory_by_default
    model = new_adapter.merge

    assert File.directory?(model.path)
  ensure
    FileUtils.remove_entry(model.path) if model
  end

  def test_merge_without_a_base_model_raises
    adapter = RPEFT::LoRA::Adapter.new(path: @adapter_dir, base_model: nil)

    assert_raises(RPEFT::MergeError) do
      adapter.merge
    end
    assert_empty @backend.calls
  end

  def test_generate_applies_the_adapter_without_merging
    completion = new_adapter.generate('User: hi')

    assert_equal 'fake completion', completion

    name, config = @backend.calls.last

    assert_equal :generate, name
    assert_equal 'fake/model', config['model']
    assert_equal @adapter_dir, config['adapter_path']
  end

  def test_chat_applies_the_adapter_to_the_transcript
    messages = [{role: 'user', content: 'hi'}]

    assert_equal 'fake completion', new_adapter.chat(messages)

    _name, config = @backend.calls.last

    assert_equal 'fake/model', config['model']
    assert_equal @adapter_dir, config['adapter_path']
    assert_equal messages, config['messages']
  end

  def test_chat_without_a_base_model_raises
    adapter = RPEFT::LoRA::Adapter.new(path: @adapter_dir, base_model: nil)

    assert_raises(RPEFT::GenerationError) do
      adapter.chat([{role: 'user', content: 'hi'}])
    end
    assert_empty @backend.calls
  end

  def test_generate_without_a_base_model_raises
    adapter = RPEFT::LoRA::Adapter.new(path: @adapter_dir, base_model: nil)

    assert_raises(RPEFT::GenerationError) do
      adapter.generate('User: hi')
    end
    assert_empty @backend.calls
  end

  def test_merge_failure_removes_its_temporary_directory
    use_backend_failing_after_partial_merge
    before = Dir.glob(File.join(Dir.tmpdir, 'rpeft-merged-*'))

    assert_raises(RPEFT::MergeError) do
      new_adapter.merge
    end
    assert_empty Dir.glob(File.join(Dir.tmpdir, 'rpeft-merged-*')) - before
  end

  def test_merge_failure_removes_the_output_dir_it_created_so_a_retry_works
    use_backend_failing_after_partial_merge
    output = File.join(@workdir, 'merged')

    assert_raises(RPEFT::MergeError) do
      new_adapter.merge(output_dir: output)
    end
    refute File.exist?(output)

    RPEFT.backend = @backend
    assert_equal output, new_adapter.merge(output_dir: output).path
  end

  def test_merge_failure_empties_but_keeps_an_existing_empty_output_dir
    use_backend_failing_after_partial_merge
    output = File.join(@workdir, 'empty')
    FileUtils.mkdir_p(output)

    assert_raises(RPEFT::MergeError) do
      new_adapter.merge(output_dir: output)
    end
    assert_empty Dir.children(output)
  end

  def test_merge_failure_keeps_a_model_merged_there_before
    output = File.join(@workdir, 'merged')
    new_adapter.merge(output_dir: output)
    use_backend_failing_after_partial_merge

    assert_raises(RPEFT::MergeError) do
      new_adapter.merge(output_dir: output)
    end
    assert_path_exists File.join(output, 'rpeft_metadata.json')
  end

  private

  # Writes part of the merged model before failing, as a merge interrupted while saving would.
  def use_backend_failing_after_partial_merge
    backend = FakeBackend.new
    backend.define_singleton_method(:merge_lora) do |config|
      File.write(File.join(config['output_dir'], 'model.safetensors'), 'partial')
      raise RPEFT::MergeError, 'disk full'
    end
    RPEFT.backend = backend
  end

  def new_adapter
    RPEFT::LoRA::Adapter.new(path: @adapter_dir, base_model: 'fake/model', rank: 8)
  end

  def write_metadata
    metadata = {
      'schema_version' => 1,
      'kind' => 'lora_adapter',
      'base_model' => 'meta/base',
      'lora' => {'rank' => 8, 'alpha' => 16, 'dropout' => 0.05,
                 'target_modules' => %w(q_proj v_proj)},
      'training' => {'train_loss' => 1.23}
    }
    File.write(File.join(@adapter_dir, 'rpeft_metadata.json'), JSON.generate(metadata))
  end
end
