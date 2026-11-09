# frozen_string_literal: true

require 'test_helper'

class TestRPEFT < RPEFTTest
  def test_that_it_has_a_version_number
    refute_nil ::RPEFT::VERSION
  end

  # Read from the file rather than imported, so the check needs no Python.
  def test_the_python_backend_has_the_same_version
    init = File.read(File.expand_path('../python/rpeft_backend/__init__.py', __dir__))

    assert_equal RPEFT::VERSION, init[/^__version__ = "(.+)"$/, 1]
  end

  def test_device_returns_the_detected_device_as_a_symbol
    assert_equal :cpu, RPEFT.device
    assert_equal :detect_device, @backend.calls.first.first
  end

  def test_device_reflects_the_backend_detection
    RPEFT.backend = FakeBackend.new(responses: {detect_device: {'device' => 'mps'}})

    assert_equal :mps, RPEFT.device
  end

  def test_check_reports_a_healthy_environment
    report = RPEFT.check

    assert_equal 'ok', report['status']
    assert_equal RPEFT::VERSION, report.dig('ruby', 'rpeft')
    assert report.dig('backend', 'ok')
    assert_equal 'mps', report.dig('device', 'recommended')
  end

  def test_check_reports_missing_libraries_as_degraded
    diagnostics = FakeBackend.healthy_diagnostics
    diagnostics['libraries']['peft'] = {'installed' => false, 'version' => nil}
    RPEFT.backend = FakeBackend.new(responses: {diagnostics: diagnostics})

    assert_equal 'degraded', RPEFT.check['status']
  end

  def test_check_reports_outdated_libraries_as_degraded
    diagnostics = FakeBackend.healthy_diagnostics
    diagnostics['libraries']['transformers']['outdated'] = true
    RPEFT.backend = FakeBackend.new(responses: {diagnostics: diagnostics})

    assert_equal 'degraded', RPEFT.check['status']
  end

  def test_check_reports_an_unusable_backend_without_raising
    RPEFT.backend = FakeBackend.new(error: RPEFT::DependencyError.new('no python'))

    report = RPEFT.check

    assert_equal 'unusable', report['status']
    refute report.dig('backend', 'ok')
    assert_includes report.dig('backend', 'error'), 'no python'
  end
end
