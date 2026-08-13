# frozen_string_literal: true

require 'test_helper'

class TestCLIGenerate < LLMTuneTest
  def test_requires_model_and_prompt
    status, _out, err = run_cli(%w(generate))

    assert_equal 2, status
    assert_includes err, '--model'
    assert_includes err, '--prompt'
    assert_empty @backend.calls
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

  def test_reports_backend_errors_with_exit_status_one
    LLMTune.backend = FakeBackend.new(error: LLMTune::GenerationError.new('generation failed'))

    status, _out, err = run_cli(['generate', '--model', 'fake/model', '--prompt', 'User: hi'])

    assert_equal 1, status
    assert_includes err, 'generation failed'
  end
end
