# frozen_string_literal: true

$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'rpeft'
require 'rpeft/cli'

require 'fileutils'
require 'json'
require 'stringio'
require 'tmpdir'

require 'minitest/autorun'

require_relative 'support/fake_backend'

# Base class for unit tests. Injects a FakeBackend so that no test
# ever loads PyCall or Python.
class RPEFTTest < Minitest::Test
  def setup
    @backend = FakeBackend.new
    RPEFT.backend = @backend
  end

  def teardown
    RPEFT.backend = nil
  end

  private

  def run_cli(argv)
    out = StringIO.new
    err = StringIO.new
    status = RPEFT::CLI.run(argv, out: out, err: err)
    [status, out.string, err.string]
  end
end
