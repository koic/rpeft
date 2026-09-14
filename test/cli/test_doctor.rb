# frozen_string_literal: true

require 'test_helper'

class TestCLIDoctor < GrainTest
  def test_reports_a_healthy_environment
    status, out, = run_cli(['doctor'])

    assert_equal 0, status
    assert_includes out, 'torch'
    assert_includes out, 'mps (CUDA: no, MPS: yes)'
    assert_includes out, 'Status:'
    assert_includes out, 'OK'
  end

  def test_reports_missing_python_packages
    diagnostics = FakeBackend.healthy_diagnostics
    diagnostics['libraries']['peft'] = {'installed' => false, 'version' => nil}
    Grain.backend = FakeBackend.new(responses: {diagnostics: diagnostics})

    status, out, = run_cli(['doctor'])

    assert_equal 1, status
    assert_includes out, 'MISSING'
    assert_includes out, 'DEGRADED'
    assert_includes out, 'pip install'
  end

  def test_reports_an_unusable_backend
    Grain.backend = FakeBackend.new(
      error: Grain::DependencyError.new('Python interpreter not found')
    )

    status, out, = run_cli(['doctor'])

    assert_equal 1, status
    assert_includes out, 'Python interpreter not found'
    assert_includes out, 'UNUSABLE'
  end
end
