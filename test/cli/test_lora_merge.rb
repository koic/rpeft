# frozen_string_literal: true

require 'test_helper'

class TestCLILoraMerge < GrainTest
  def setup
    super
    @workdir = Dir.mktmpdir('grain_test')
    @adapter_dir = File.join(@workdir, 'adapter')
    @merged_dir = File.join(@workdir, 'merged')
    FileUtils.mkdir_p(@adapter_dir)
  end

  def teardown
    FileUtils.remove_entry(@workdir)
    super
  end

  def test_requires_adapter_and_output
    status, _out, err = run_cli(%w(lora merge))

    assert_equal 2, status
    assert_includes err, '--adapter'
    assert_includes err, '--output'
  end

  def test_merges_using_the_recorded_base_model
    write_metadata(base_model: 'meta/base')

    status, out, = run_cli(['lora', 'merge', '--adapter', @adapter_dir, '--output', @merged_dir])

    assert_equal 0, status
    assert_includes out, "Merged model saved to #{@merged_dir}"

    name, config = @backend.calls.last
    assert_equal :merge_lora, name
    assert_equal @adapter_dir, config['adapter_path']
    assert_equal @merged_dir, config['output_dir']
    assert_equal 'meta/base', config['base_model']
  end

  def test_model_option_overrides_the_recorded_base_model
    write_metadata(base_model: 'meta/base')

    status, = run_cli(['lora', 'merge', '--adapter', @adapter_dir, '--output', @merged_dir,
                       '--model', 'override/base'])

    assert_equal 0, status
    _name, config = @backend.calls.last
    assert_equal 'override/base', config['base_model']
  end

  def test_fails_for_a_directory_that_is_not_an_adapter
    status, _out, err = run_cli(['lora', 'merge', '--adapter', @adapter_dir,
                                 '--output', @merged_dir])

    assert_equal 1, status
    assert_includes err, 'not a LoRA adapter directory'
  end

  private

  def write_metadata(base_model:)
    metadata = {
      'schema_version' => 1,
      'kind' => 'lora_adapter',
      'base_model' => base_model,
      'lora' => {'rank' => 8}
    }
    File.write(File.join(@adapter_dir, 'grain.json'), JSON.generate(metadata))
  end
end
