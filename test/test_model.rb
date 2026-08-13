# frozen_string_literal: true

require 'test_helper'

class TestModel < LLMTuneTest
  def setup
    super
    @workdir = Dir.mktmpdir('llm_tune_test')
    @model_dir = File.join(@workdir, 'model')
    FileUtils.mkdir_p(@model_dir)
    File.write(File.join(@model_dir, 'config.json'), '{}')
  end

  def teardown
    FileUtils.remove_entry(@workdir)
    super
  end

  def test_exposes_its_path_and_base_model
    model = LLMTune::Model.new(path: @model_dir, base_model: 'fake/model')

    assert_equal @model_dir, model.path
    assert_equal 'fake/model', model.base_model
  end

  def test_save_moves_the_model_directory
    model = LLMTune::Model.new(path: @model_dir)
    dest = File.join(@workdir, 'moved')

    assert_same model, model.save(dest)
    assert_equal dest, model.path
    assert_path_exists File.join(dest, 'config.json')
    refute File.exist?(@model_dir)
  end

  def test_save_refuses_to_overwrite_an_unrelated_existing_directory
    dest = File.join(@workdir, 'existing')
    FileUtils.mkdir_p(dest)
    model = LLMTune::Model.new(path: @model_dir)

    assert_raises(LLMTune::Error) { model.save(dest) }
  end

  def test_save_replaces_a_previously_saved_model
    dest = File.join(@workdir, 'saved')
    FileUtils.mkdir_p(dest)
    File.write(File.join(dest, 'config.json'), '{}')
    File.write(File.join(dest, 'stale'), 'old')
    model = LLMTune::Model.new(path: @model_dir)

    model.save(dest)

    assert_path_exists File.join(dest, 'config.json')
    refute File.exist?(File.join(dest, 'stale'))
  end

  def test_keeps_a_hub_model_id_verbatim
    model = LLMTune::Model.new(path: 'Qwen/Qwen2.5-Coder-0.5B')

    assert_equal 'Qwen/Qwen2.5-Coder-0.5B', model.path
  end

  def test_generate_returns_the_completion_from_the_backend
    model = LLMTune::Model.new(path: @model_dir)

    assert_equal 'fake completion', model.generate('User: hi')
  end

  def test_generate_sends_the_generation_configuration_to_the_backend
    LLMTune::Model.new(path: @model_dir).generate('User: hi', max_new_tokens: 8)

    name, config = @backend.calls.last

    assert_equal :generate, name
    assert_equal @model_dir, config['model']
    assert_equal 'User: hi', config['prompt']
    assert_equal 8, config['max_new_tokens']
    assert_equal 'cpu', config['device']
  end

  def test_generate_passes_an_explicit_device_through_without_detection
    LLMTune::Model.new(path: @model_dir).generate('User: hi', device: :mps)

    refute(@backend.calls.any? { |call| call.first == :detect_device })
    _name, config = @backend.calls.last
    assert_equal 'mps', config['device']
  end

  def test_generate_rejects_a_non_positive_max_new_tokens
    model = LLMTune::Model.new(path: @model_dir)

    assert_raises(LLMTune::ConfigurationError) { model.generate('User: hi', max_new_tokens: 0) }
  end

  def test_generate_rejects_an_unknown_device
    model = LLMTune::Model.new(path: @model_dir)

    assert_raises(LLMTune::ConfigurationError) { model.generate('User: hi', device: :tpu) }
  end
end
