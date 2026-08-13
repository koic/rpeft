# frozen_string_literal: true

require 'test_helper'

class TestLLMTune < LLMTuneTest
  def test_that_it_has_a_version_number
    refute_nil ::LLMTune::VERSION
  end

  def test_device_returns_the_detected_device_as_a_symbol
    assert_equal :cpu, LLMTune.device
    assert_equal :detect_device, @backend.calls.first.first
  end

  def test_device_reflects_the_backend_detection
    LLMTune.backend = FakeBackend.new(responses: {detect_device: {'device' => 'mps'}})

    assert_equal :mps, LLMTune.device
  end

  def test_check_reports_a_healthy_environment
    report = LLMTune.check

    assert_equal 'ok', report['status']
    assert_equal LLMTune::VERSION, report.dig('ruby', 'llm_tune')
    assert report.dig('backend', 'ok')
    assert_equal 'mps', report.dig('device', 'recommended')
  end

  def test_check_reports_missing_libraries_as_degraded
    diagnostics = FakeBackend.healthy_diagnostics
    diagnostics['libraries']['peft'] = {'installed' => false, 'version' => nil}
    LLMTune.backend = FakeBackend.new(responses: {diagnostics: diagnostics})

    assert_equal 'degraded', LLMTune.check['status']
  end

  def test_check_reports_an_unusable_backend_without_raising
    LLMTune.backend = FakeBackend.new(error: LLMTune::DependencyError.new('no python'))

    report = LLMTune.check

    assert_equal 'unusable', report['status']
    refute report.dig('backend', 'ok')
    assert_includes report.dig('backend', 'error'), 'no python'
  end
end
