# frozen_string_literal: true

require 'test_helper'

class TestGemspec < Minitest::Test
  GEMSPEC = File.expand_path('../rpeft.gemspec', __dir__)

  def test_lists_the_files_to_ship_wherever_the_gemspec_is_loaded_from
    spec = Dir.chdir(Dir.tmpdir) { load_gemspec }

    assert_includes spec.files, 'lib/rpeft.rb'
    assert_includes spec.files, 'exe/rpeft'
    assert_includes spec.files, 'python/rpeft_backend/__init__.py'
    assert_equal %w(rpeft), spec.executables
  end

  def test_links_each_uri_once
    spec = load_gemspec
    uris = spec.metadata.select { |key, _| key.end_with?('_uri') }.values

    assert_equal uris.uniq, uris
  end

  private

  # Evaluated directly rather than through Gem::Specification.load, which caches by path.
  def load_gemspec
    eval(File.read(GEMSPEC), binding, GEMSPEC) # rubocop:disable Security/Eval
  end
end
