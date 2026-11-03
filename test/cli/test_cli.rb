# frozen_string_literal: true

require 'test_helper'

class TestCLI < RPEFTTest
  def test_prints_usage_without_arguments
    status, out, = run_cli([])

    assert_equal 0, status
    assert_includes out, 'Usage: rpeft'
  end

  def test_prints_usage_for_help
    status, out, = run_cli(['help'])

    assert_equal 0, status
    assert_includes out, 'Usage: rpeft'
  end

  def test_prints_the_version
    status, out, = run_cli(['version'])

    assert_equal 0, status
    assert_equal "#{RPEFT::VERSION}\n", out
  end

  def test_rejects_an_unknown_command
    status, _out, err = run_cli(['bogus'])

    assert_equal 2, status
    assert_includes err, 'unknown command'
  end

  def test_lists_lora_subcommands
    status, out, = run_cli(['lora'])

    assert_equal 0, status
    assert_includes out, 'train'
    assert_includes out, 'merge'
  end

  def test_rejects_an_unknown_lora_subcommand
    status, _out, err = run_cli(%w(lora bogus))

    assert_equal 2, status
    assert_includes err, 'unknown lora subcommand'
  end

  def test_prints_command_help
    status, out, = run_cli(%w(lora train --help))

    assert_equal 0, status
    assert_includes out, '--rank'
  end
end
