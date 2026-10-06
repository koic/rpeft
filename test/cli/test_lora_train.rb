# frozen_string_literal: true

require 'test_helper'

class TestCLILoraTrain < GrainTest
  def setup
    super
    @workdir = Dir.mktmpdir('grain_test')
    @dataset = File.join(@workdir, 'train.jsonl')
    File.write(@dataset, %({"text":"User: hi\\nAssistant: hello"}\n))
  end

  def teardown
    FileUtils.remove_entry(@workdir)
    super
  end

  def test_requires_model_dataset_and_output
    status, _out, err = run_cli(%w(lora train))

    assert_equal 2, status
    assert_includes err, '--model'
    assert_includes err, '--dataset'
    assert_includes err, '--output'
    assert_empty @backend.calls
  end

  def test_help_lists_every_dataset_format
    status, out, = run_cli(%w(lora train --help))

    assert_equal 0, status
    assert_includes out, '"text"'
    assert_includes out, '"instruction"/"output"'
    assert_includes out, '"messages"'
  end

  def test_rejects_an_invalid_option_value
    status, _out, err = run_cli(['lora', 'train', '--model', 'm', '--dataset', @dataset,
                                 '--output', File.join(@workdir, 'adapter'), '--rank', 'x'])

    assert_equal 2, status
    assert_includes err, '--rank'
  end

  def test_trains_and_reports_the_adapter_path
    output = File.join(@workdir, 'adapter')

    status, out, = run_cli(['lora', 'train', '--model', 'fake/model', '--dataset', @dataset,
                            '--output', output, '--rank', '2', '--epochs', '1',
                            '--target-modules', 'q_proj,v_proj', '--device', 'cpu'])

    assert_equal 0, status
    assert_includes out, "Adapter saved to #{output}"
    assert_includes out, 'Final training loss:'

    _name, config = @backend.calls.find { |call| call.first == :train_lora }
    assert_equal 'fake/model', config['base_model']
    assert_equal 2, config.dig('lora', 'r')
    assert_equal %w(q_proj v_proj), config.dig('lora', 'target_modules')
    assert_equal 1, config['epochs']
    assert_equal 'cpu', config['device']
  end

  def test_reports_backend_errors_with_exit_status_one
    Grain.backend = FakeBackend.new(error: Grain::TrainingError.new('loss exploded'))

    status, _out, err = run_cli(['lora', 'train', '--model', 'm', '--dataset', @dataset,
                                 '--output', File.join(@workdir, 'adapter')])

    assert_equal 1, status
    assert_includes err, 'loss exploded'
  end
end
