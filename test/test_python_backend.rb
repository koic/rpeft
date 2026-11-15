# frozen_string_literal: true

require 'test_helper'
require 'rpeft/backend/python'

class TestPythonBackend < Minitest::Test
  # Stands in for Python's atexit module.
  class FakeAtexit
    attr_reader :ran

    def _run_exitfuncs
      @ran = true
    end
  end

  # Stands in for the Python side of train_lora: it calls the progress callback as Python would, where
  # an exception raised by Ruby is lost, and gives up the run when the callback answers "stop".
  class TrainingBackend < RPEFT::Backend::Python
    attr_reader :answers

    private

    def call(_name, _config, callback = nil, default_error:)
      @answers = Array.new(3) { |step|
        begin
          callback&.call(%({"step": #{step}}))
        rescue Exception # rubocop:disable Lint/RescueException
          nil
        end
      }
      raise default_error, 'training was stopped by the progress callback' if @answers.include?('stop')

      {'steps' => 3}
    end
  end

  def setup
    @workdir = Dir.mktmpdir('rpeft_test')
  end

  def teardown
    FileUtils.remove_entry(@workdir)
  end

  def test_train_lora_yields_each_progress_entry_as_a_hash
    backend = TrainingBackend.new
    seen = []

    result = backend.train_lora({}) { |entry| seen << entry }

    assert_equal [{'step' => 0}, {'step' => 1}, {'step' => 2}], seen
    assert_equal({'steps' => 3}, result)
    assert_equal [nil, nil, nil], backend.answers
  end

  def test_train_lora_stops_the_run_and_raises_what_the_block_raised
    backend = TrainingBackend.new
    calls = 0

    error = assert_raises(ArgumentError) {
      backend.train_lora({}) do |_entry|
        calls += 1
        raise ArgumentError, 'from the block'
      end
    }

    assert_equal 'from the block', error.message
    assert_equal 1, calls
    assert_equal %w(stop stop stop), backend.answers
  end

  def test_train_lora_stops_the_run_on_an_interrupt
    backend = TrainingBackend.new

    assert_raises(Interrupt) do
      backend.train_lora({}) { |_entry| raise Interrupt }
    end
    assert_includes backend.answers, 'stop'
  end

  def test_train_lora_stops_the_run_when_the_block_is_left_with_break
    backend = TrainingBackend.new
    calls = 0

    error = assert_raises(RPEFT::TrainingError) {
      backend.train_lora({}) do |_entry|
        calls += 1
        break
      end
    }

    assert_match(/break or throw/, error.message)
    assert_equal 1, calls
    assert_equal %w(stop stop stop), backend.answers
  end

  def test_train_lora_stops_the_run_when_the_block_is_left_with_throw
    backend = TrainingBackend.new

    error = assert_raises(RPEFT::TrainingError) {
      catch(:done) do
        backend.train_lora({}) { |_entry| throw :done }
      end
    }

    assert_match(/break or throw/, error.message)
    assert_equal %w(stop stop stop), backend.answers
  end

  def test_train_lora_without_a_block_passes_no_callback
    backend = TrainingBackend.new

    assert_equal({'steps' => 3}, backend.train_lora({}))
    assert_equal [nil, nil, nil], backend.answers
  end

  def test_venv_site_packages_finds_the_site_packages_directory
    python = fake_venv_python(with_pyvenv_cfg: true)
    expected = File.join(@workdir, 'venv', 'lib', 'python3.12', 'site-packages')

    assert_equal expected, RPEFT::Backend::Python.venv_site_packages(python)
  end

  def test_venv_site_packages_returns_nil_without_pyvenv_cfg
    python = fake_venv_python(with_pyvenv_cfg: false)

    assert_nil RPEFT::Backend::Python.venv_site_packages(python)
  end

  def test_venv_site_packages_returns_nil_for_nil_path
    assert_nil RPEFT::Backend::Python.venv_site_packages(nil)
  end

  def test_default_python_prefers_the_python_environment_variable
    with_env('PYTHON' => '/opt/custom/python') do
      assert_equal '/opt/custom/python', RPEFT::Backend::Python.default_python
    end
  end

  def test_default_python_falls_back_to_the_venv_next_to_the_gemfile
    fake_dot_venv
    gemfile = File.join(@workdir, 'Gemfile')
    subdir = File.join(@workdir, 'scripts')
    FileUtils.mkdir_p(subdir)

    with_env('PYTHON' => nil, 'BUNDLE_GEMFILE' => gemfile) do
      Dir.chdir(subdir) do
        assert_equal File.join(@workdir, '.venv', 'bin', 'python'), RPEFT::Backend::Python.default_python
      end
    end
  end

  def test_default_python_ignores_a_venv_in_the_current_directory_outside_a_bundle
    fake_dot_venv

    with_env('PYTHON' => nil, 'BUNDLE_GEMFILE' => nil) do
      Dir.chdir(@workdir) do
        assert_nil RPEFT::Backend::Python.default_python
      end
    end
  end

  def test_default_python_returns_nil_without_a_venv_or_environment_variable
    with_env('PYTHON' => nil, 'BUNDLE_GEMFILE' => File.join(@workdir, 'Gemfile')) do
      assert_nil RPEFT::Backend::Python.default_python
    end
  end

  def test_run_exit_handlers_runs_the_python_atexit_handlers
    atexit = FakeAtexit.new

    RPEFT::Backend::Python.run_exit_handlers(atexit)

    assert atexit.ran
  end

  def test_run_exit_handlers_ignores_a_failing_python_handler
    atexit = Object.new
    def atexit._run_exitfuncs
      raise 'handler failed'
    end

    assert_nil RPEFT::Backend::Python.run_exit_handlers(atexit)
  end

  def test_install_exit_hook_runs_the_python_atexit_handlers_at_exit
    backend = RPEFT::Backend::Python.new
    hooks = []
    backend.define_singleton_method(:at_exit) do |&hook|
      hooks << hook
    end
    atexit = FakeAtexit.new

    backend.send(:install_exit_hook, atexit)

    refute atexit.ran
    hooks.each(&:call)
    assert atexit.ran
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

  def fake_dot_venv
    python = fake_venv_python(with_pyvenv_cfg: true)
    FileUtils.chmod(0o755, python)
    FileUtils.mv(File.join(@workdir, 'venv'), File.join(@workdir, '.venv'))
  end
end
