# frozen_string_literal: true

require 'test_helper'

class TestModel < RPEFTTest
  def setup
    super
    @workdir = Dir.mktmpdir('rpeft_test')
    @model_dir = File.join(@workdir, 'model')
    FileUtils.mkdir_p(@model_dir)
    File.write(File.join(@model_dir, 'config.json'), '{}')
  end

  def teardown
    FileUtils.remove_entry(@workdir)
    super
  end

  def test_exposes_its_path_and_base_model
    model = RPEFT::Model.new(path: @model_dir, base_model: 'fake/model')

    assert_equal @model_dir, model.path
    assert_equal 'fake/model', model.base_model
  end

  def test_save_moves_the_model_directory
    model = RPEFT::Model.new(path: @model_dir)
    dest = File.join(@workdir, 'moved')

    assert_same model, model.save(dest)
    assert_equal dest, model.path
    assert_path_exists File.join(dest, 'config.json')
    refute File.exist?(@model_dir)
  end

  def test_save_refuses_to_overwrite_an_unrelated_existing_directory
    dest = File.join(@workdir, 'existing')
    FileUtils.mkdir_p(dest)
    model = RPEFT::Model.new(path: @model_dir)

    assert_raises(RPEFT::ConfigurationError) { model.save(dest) }
  end

  def test_save_replaces_a_previously_saved_model
    dest = File.join(@workdir, 'saved')
    FileUtils.mkdir_p(dest)
    RPEFT::Model.write_metadata(dest, base_model: 'fake/model')
    File.write(File.join(dest, 'stale'), 'old')
    model = RPEFT::Model.new(path: @model_dir)

    model.save(dest)

    assert_path_exists File.join(dest, 'config.json')
    refute File.exist?(File.join(dest, 'stale'))
  end

  def test_save_of_a_hub_model_keeps_the_previously_saved_model
    dest = File.join(@workdir, 'saved')
    FileUtils.mkdir_p(dest)
    RPEFT::Model.write_metadata(dest, base_model: 'fake/model')
    model = RPEFT::Model.new(path: 'Qwen/Qwen2.5-Coder-0.5B')

    error = assert_raises(RPEFT::Error) { model.save(dest) }

    assert_match(/not a local directory/, error.message)
    assert_path_exists File.join(dest, 'rpeft_metadata.json')
  end

  def test_save_to_its_own_path_keeps_the_model
    RPEFT::Model.write_metadata(@model_dir, base_model: 'fake/model')
    model = RPEFT::Model.new(path: @model_dir)

    model.save(@model_dir)

    assert_equal @model_dir, model.path
    assert_path_exists File.join(@model_dir, 'config.json')
  end

  def test_save_refuses_to_overwrite_a_directory_that_only_has_a_config_json
    dest = File.join(@workdir, 'app')
    FileUtils.mkdir_p(dest)
    File.write(File.join(dest, 'config.json'), '{}')
    model = RPEFT::Model.new(path: @model_dir)

    assert_raises(RPEFT::ConfigurationError) do
      model.save(dest)
    end
    assert_path_exists File.join(dest, 'config.json')
  end

  def test_write_metadata_marks_a_merged_model
    RPEFT::Model.write_metadata(@model_dir, base_model: 'fake/model')

    metadata = JSON.parse(File.read(File.join(@model_dir, 'rpeft_metadata.json')))

    assert_equal 'merged_model', metadata['kind']
    assert_equal 'fake/model', metadata['base_model']
  end

  def test_keeps_a_hub_model_id_verbatim
    model = RPEFT::Model.new(path: 'Qwen/Qwen2.5-Coder-0.5B')

    assert_equal 'Qwen/Qwen2.5-Coder-0.5B', model.path
  end

  def test_generate_returns_the_completion_from_the_backend
    model = RPEFT::Model.new(path: @model_dir)

    assert_equal 'fake completion', model.generate('User: hi')
  end

  def test_generate_sends_the_generation_configuration_to_the_backend
    RPEFT::Model.new(path: @model_dir).generate('User: hi', max_new_tokens: 8)

    name, config = @backend.calls.last

    assert_equal :generate, name
    assert_equal @model_dir, config['model']
    assert_equal 'User: hi', config['prompt']
    assert_equal 8, config['max_new_tokens']
    assert_equal 'cpu', config['device']
  end

  def test_generate_passes_an_explicit_device_through_without_detection
    RPEFT::Model.new(path: @model_dir).generate('User: hi', device: :mps)

    refute(@backend.calls.any? { |call| call.first == :detect_device })
    _name, config = @backend.calls.last
    assert_equal 'mps', config['device']
  end

  def test_generate_rejects_a_non_positive_max_new_tokens
    model = RPEFT::Model.new(path: @model_dir)

    assert_raises(RPEFT::ConfigurationError) { model.generate('User: hi', max_new_tokens: 0) }
  end

  def test_generate_rejects_an_unknown_device
    model = RPEFT::Model.new(path: @model_dir)

    assert_raises(RPEFT::ConfigurationError) { model.generate('User: hi', device: :tpu) }
  end

  def test_generate_passes_sampling_options_when_given
    RPEFT::Model.new(path: @model_dir).generate('User: hi', temperature: 0.7, top_p: 0.9, seed: 42)

    _name, config = @backend.calls.last

    assert_in_delta 0.7, config['temperature']
    assert_in_delta 0.9, config['top_p']
    assert_equal 42, config['seed']
  end

  def test_generate_omits_sampling_options_by_default
    RPEFT::Model.new(path: @model_dir).generate('User: hi')

    _name, config = @backend.calls.last

    refute config.key?('temperature')
    refute config.key?('top_p')
    refute config.key?('seed')
  end

  def test_generate_rejects_a_non_positive_temperature
    model = RPEFT::Model.new(path: @model_dir)

    assert_raises(RPEFT::ConfigurationError) { model.generate('User: hi', temperature: 0) }
  end

  def test_generate_rejects_an_out_of_range_top_p
    model = RPEFT::Model.new(path: @model_dir)

    assert_raises(RPEFT::ConfigurationError) { model.generate('User: hi', top_p: 1.5) }
  end

  def test_chat_returns_the_completion_from_the_backend
    model = RPEFT::Model.new(path: @model_dir)

    assert_equal 'fake completion', model.chat([{role: 'user', content: 'hi'}])
  end

  def test_chat_sends_messages_instead_of_a_prompt
    messages = [{role: 'user', content: 'hi'}]

    RPEFT::Model.new(path: @model_dir).chat(messages, max_new_tokens: 8, device: :cpu)

    name, config = @backend.calls.last

    assert_equal :generate, name
    assert_equal messages, config['messages']
    refute config.key?('prompt')
    assert_equal 8, config['max_new_tokens']
    assert_equal 'cpu', config['device']
  end

  def test_chat_accepts_string_keyed_messages
    RPEFT::Model.new(path: @model_dir).chat([{'role' => 'user', 'content' => 'hi'}])

    _name, config = @backend.calls.last

    assert_equal 'user', config.dig('messages', 0, 'role')
  end

  def test_chat_rejects_messages_that_are_not_an_array
    model = RPEFT::Model.new(path: @model_dir)

    assert_raises(RPEFT::ConfigurationError) do
      model.chat('hi')
    end
    assert_empty @backend.calls
  end

  def test_rejects_a_missing_path
    assert_raises(RPEFT::ConfigurationError) { RPEFT::Model.new(path: nil) }
  end

  def test_accepts_a_pathname
    assert_equal @model_dir, RPEFT::Model.new(path: Pathname(@model_dir)).path
  end

  def test_save_rejects_a_missing_destination
    model = RPEFT::Model.new(path: @model_dir)

    assert_raises(RPEFT::ConfigurationError) do
      model.save(nil)
    end
    assert_path_exists @model_dir
  end

  def test_generate_rejects_a_missing_prompt_before_calling_the_backend
    model = RPEFT::Model.new(path: @model_dir)

    assert_raises(RPEFT::ConfigurationError) do
      model.generate(nil)
    end
    assert_raises(RPEFT::ConfigurationError) do
      model.generate('')
    end
    assert_raises(RPEFT::ConfigurationError) do
      model.generate(42)
    end
    assert_empty @backend.calls
  end

  def test_generate_rejects_a_nil_device_as_a_configuration_error
    model = RPEFT::Model.new(path: @model_dir)

    assert_raises(RPEFT::ConfigurationError) { model.generate('User: hi', device: nil) }
  end

  def test_generate_accepts_a_device_given_as_a_string
    RPEFT::Model.new(path: @model_dir).generate('User: hi', device: 'mps')

    assert_equal 'mps', @backend.calls.last.last['device']
  end

  def test_chat_accepts_the_system_user_and_assistant_roles
    model = RPEFT::Model.new(path: @model_dir)

    model.chat(%w(system user assistant).map { |role| {role: role, content: 'hi'} })

    assert_equal(%w(system user assistant), @backend.calls.last.last['messages'].map { |message| message[:role] })
  end

  def test_chat_rejects_an_unknown_role_before_calling_the_backend
    model = RPEFT::Model.new(path: @model_dir)

    assert_raises(RPEFT::ConfigurationError) do
      model.chat([{role: 'tool', content: 'hi'}])
    end
    assert_empty @backend.calls
  end

  def test_chat_rejects_messages_without_role_and_content
    model = RPEFT::Model.new(path: @model_dir)

    assert_raises(RPEFT::ConfigurationError) { model.chat([{speaker: 'user', text: 'hi'}]) }
  end

  def test_chat_rejects_an_empty_transcript
    model = RPEFT::Model.new(path: @model_dir)

    assert_raises(RPEFT::ConfigurationError) { model.chat([]) }
  end

  def test_generate_rejects_a_non_integer_seed
    model = RPEFT::Model.new(path: @model_dir)

    assert_raises(RPEFT::ConfigurationError) { model.generate('User: hi', seed: 'x') }
  end
end
