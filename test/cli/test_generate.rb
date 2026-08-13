# frozen_string_literal: true

require 'test_helper'

class TestCLIGenerate < LLMTuneTest
  def setup
    super
    @workdir = Dir.mktmpdir('llm_tune_test')
  end

  def teardown
    FileUtils.remove_entry(@workdir)
    super
  end

  def test_requires_a_prompt
    status, _out, err = run_cli(%w(generate))

    assert_equal 2, status
    assert_includes err, '--prompt'
    assert_empty @backend.calls
  end

  def test_requires_a_model_or_an_adapter
    status, _out, err = run_cli(['generate', '--prompt', 'User: hi'])

    assert_equal 2, status
    assert_includes err, '--model or --adapter'
    assert_empty @backend.calls
  end

  def test_generates_from_an_adapter_without_merging
    adapter_dir = File.join(@workdir, 'adapter')
    FileUtils.mkdir_p(adapter_dir)
    metadata = {'kind' => 'lora_adapter', 'base_model' => 'meta/base', 'lora' => {'rank' => 8}}
    File.write(File.join(adapter_dir, 'llm_tune.json'), JSON.generate(metadata))

    status, out, = run_cli(['generate', '--adapter', adapter_dir, '--prompt', 'User: hi', '--device', 'cpu'])

    assert_equal 0, status
    assert_includes out, 'fake completion'

    _name, config = @backend.calls.last

    assert_equal 'meta/base', config['model']
    assert_equal adapter_dir, config['adapter_path']
  end

  def test_prints_the_completion
    status, out, = run_cli(['generate', '--model', 'fake/model', '--prompt', 'User: hi', '--max-new-tokens', '8', '--device', 'cpu'])

    assert_equal 0, status
    assert_includes out, 'fake completion'

    name, config = @backend.calls.last

    assert_equal :generate, name
    assert_equal 'fake/model', config['model']
    assert_equal 'User: hi', config['prompt']
    assert_equal 8, config['max_new_tokens']
    assert_equal 'cpu', config['device']
  end

  def test_passes_sampling_options_through
    status, = run_cli(['generate', '--model', 'fake/model', '--prompt', 'User: hi', '--temperature', '0.7', '--top-p', '0.9', '--seed', '42'])

    assert_equal 0, status

    _name, config = @backend.calls.last

    assert_in_delta 0.7, config['temperature']
    assert_in_delta 0.9, config['top_p']
    assert_equal 42, config['seed']
  end

  def test_reports_backend_errors_with_exit_status_one
    LLMTune.backend = FakeBackend.new(error: LLMTune::GenerationError.new('generation failed'))

    status, _out, err = run_cli(['generate', '--model', 'fake/model', '--prompt', 'User: hi'])

    assert_equal 1, status
    assert_includes err, 'generation failed'
  end
end
