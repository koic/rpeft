# frozen_string_literal: true

require 'test_helper'

class TestCLIDoctor < RPEFTTest
  def test_reports_a_healthy_environment
    status, out, = run_cli(['doctor'])

    assert_equal 0, status
    assert_includes out, 'torch'
    assert_includes out, 'mps (CUDA: no, MPS: yes)'
    assert_includes out, 'Status:'
    assert_includes out, 'OK'
  end

  def test_reports_the_optional_llama_cpp_tools_without_affecting_the_status
    env = save_env('LLAMA_CPP_DIR', 'PATH')
    ENV.delete('LLAMA_CPP_DIR')
    ENV['PATH'] = ''

    status, out, = run_cli(['doctor'])

    assert_equal 0, status
    assert_includes out, 'llama.cpp:    not configured (optional, set LLAMA_CPP_DIR for GGUF export)'
    assert_includes out, 'Quantizer:    no llama-quantize'
  ensure
    restore_env(env)
  end

  def test_reports_a_llama_cpp_dir_without_the_converter
    env = save_env('LLAMA_CPP_DIR')
    Dir.mktmpdir('rpeft_test') do |dir|
      ENV['LLAMA_CPP_DIR'] = dir

      _status, out, = run_cli(['doctor'])

      assert_includes out, "llama.cpp:    NG no convert_hf_to_gguf.py in #{dir}"
    end
  ensure
    restore_env(env)
  end

  def test_escapes_control_characters_from_the_environment
    env = save_env('LLAMA_CPP_DIR')
    ENV['LLAMA_CPP_DIR'] = "/tmp/llama\e]52;c;Zm9v\a.cpp"

    _status, out, = run_cli(['doctor'])

    refute_includes out, "\e"
    assert_includes out, 'llama\\x1B]52;c;Zm9v\\x07.cpp'
  ensure
    restore_env(env)
  end

  def test_keeps_running_with_a_llama_cpp_dir_that_cannot_be_expanded
    env = save_env('LLAMA_CPP_DIR')
    ENV['LLAMA_CPP_DIR'] = '~rpeft_no_such_user/llama.cpp'

    status, out, = run_cli(['doctor'])

    assert_equal 0, status
    assert_includes out, 'llama.cpp:    NG no convert_hf_to_gguf.py in ~rpeft_no_such_user/llama.cpp'
  ensure
    restore_env(env)
  end

  def test_reports_missing_python_packages
    diagnostics = FakeBackend.healthy_diagnostics
    diagnostics['libraries']['peft'] = {'installed' => false, 'version' => nil}
    RPEFT.backend = FakeBackend.new(responses: {diagnostics: diagnostics})

    status, out, = run_cli(['doctor'])

    assert_equal 1, status
    assert_includes out, 'MISSING'
    assert_includes out, 'DEGRADED'
    assert_includes out, 'pip install'
  end

  def test_reports_outdated_python_packages
    diagnostics = FakeBackend.healthy_diagnostics
    diagnostics['libraries']['torch'] = {'installed' => true, 'version' => '2.5.1', 'minimum' => '2.6', 'outdated' => true}
    RPEFT.backend = FakeBackend.new(responses: {diagnostics: diagnostics})

    status, out, = run_cli(['doctor'])

    assert_equal 1, status
    assert_includes out, 'OUTDATED 2.5.1 (needs 2.6 or later)'
    assert_includes out, 'DEGRADED'
    assert_includes out, 'pip install --upgrade'
  end

  def test_reports_an_unusable_backend
    RPEFT.backend = FakeBackend.new(
      error: RPEFT::DependencyError.new('Python interpreter not found')
    )

    status, out, = run_cli(['doctor'])

    assert_equal 1, status
    assert_includes out, 'Python interpreter not found'
    assert_includes out, 'UNUSABLE'
  end
end
