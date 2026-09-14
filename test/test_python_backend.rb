# frozen_string_literal: true

require 'test_helper'
require 'grain/backend/python'

class TestPythonBackend < Minitest::Test
  def setup
    @workdir = Dir.mktmpdir('grain_test')
  end

  def teardown
    FileUtils.remove_entry(@workdir)
  end

  def test_venv_site_packages_finds_the_site_packages_directory
    python = fake_venv_python(with_pyvenv_cfg: true)
    expected = File.join(@workdir, 'venv', 'lib', 'python3.12', 'site-packages')

    assert_equal expected, Grain::Backend::Python.venv_site_packages(python)
  end

  def test_venv_site_packages_returns_nil_without_pyvenv_cfg
    python = fake_venv_python(with_pyvenv_cfg: false)

    assert_nil Grain::Backend::Python.venv_site_packages(python)
  end

  def test_venv_site_packages_returns_nil_for_nil_path
    assert_nil Grain::Backend::Python.venv_site_packages(nil)
  end

  def test_default_python_prefers_the_python_environment_variable
    with_env('PYTHON' => '/opt/custom/python') do
      assert_equal '/opt/custom/python', Grain::Backend::Python.default_python
    end
  end

  def test_default_python_falls_back_to_a_local_venv
    python = fake_venv_python(with_pyvenv_cfg: true)
    FileUtils.chmod(0o755, python)
    FileUtils.mv(File.join(@workdir, 'venv'), File.join(@workdir, '.venv'))

    with_env('PYTHON' => nil) do
      Dir.chdir(@workdir) do
        expected = File.expand_path('.venv/bin/python', Dir.pwd)

        assert_equal expected, Grain::Backend::Python.default_python
      end
    end
  end

  def test_default_python_returns_nil_without_a_venv_or_environment_variable
    with_env('PYTHON' => nil) do
      Dir.chdir(@workdir) do
        assert_nil Grain::Backend::Python.default_python
      end
    end
  end

  private

  def with_env(pairs)
    saved = pairs.keys.to_h { |key| [key, ENV.fetch(key, nil)] }
    pairs.each do |key, value|
      ENV[key] = value
    end
    yield
  ensure
    saved.each { |key, value| ENV[key] = value }
  end

  def fake_venv_python(with_pyvenv_cfg:)
    venv = File.join(@workdir, 'venv')
    FileUtils.mkdir_p(File.join(venv, 'bin'))
    FileUtils.mkdir_p(File.join(venv, 'lib', 'python3.12', 'site-packages'))
    python = File.join(venv, 'bin', 'python')
    File.write(python, '')
    File.write(File.join(venv, 'pyvenv.cfg'), "home = /opt/python\n") if with_pyvenv_cfg
    python
  end
end
