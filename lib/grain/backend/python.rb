# frozen_string_literal: true

require 'json'

module Grain
  module Backend
    # Default backend. Bridges to the bundled python/grain_backend package through PyCall.
    # Only JSON strings cross the PyCall boundary, so no Python object ever leaks into the Ruby API.
    #
    # PyCall and the Python interpreter are loaded lazily on the first call: `require "grain"`
    # must keep working without a Python environment.
    class Python
      PYTHON_DIR = File.expand_path('../../../python', __dir__)

      # The interpreter to use: the PYTHON environment variable when set, otherwise .venv/bin/python next to
      # the Gemfile of the running bundle, otherwise nil (PyCall falls back to python3 on PATH).
      # The bundle root rather than the current directory, so the choice does not depend on where a command
      # happens to be run.
      def self.default_python
        python = ENV.fetch('PYTHON', nil)
        return python if python && !python.empty?

        gemfile = ENV.fetch('BUNDLE_GEMFILE', nil)
        return nil if gemfile.nil? || gemfile.empty?

        bundle_venv = File.join(File.dirname(File.expand_path(gemfile)), '.venv', 'bin', 'python')

        File.executable?(bundle_venv) ? bundle_venv : nil
      end

      # When PYTHON points into a venv, returns its site-packages directory; nil otherwise.
      # Embedded CPython reports the Ruby binary as sys.executable, so the interpreter never detects
      # pyvenv.cfg on its own and the venv packages would be invisible without this.
      def self.venv_site_packages(python_path)
        return nil unless python_path

        venv_root = File.dirname(File.dirname(File.expand_path(python_path)))
        return nil unless File.file?(File.join(venv_root, 'pyvenv.cfg'))

        Dir[File.join(venv_root, 'lib', 'python*', 'site-packages')].first
      end

      # PyCall never finalizes the embedded interpreter, so without this Python's atexit handlers never run.
      # multiprocessing relies on them to unregister its semaphores (tqdm creates one for its progress bars),
      # and resource_tracker would otherwise warn about leaked semaphores when the process exits.
      def self.run_exit_handlers(atexit)
        atexit._run_exitfuncs
      rescue StandardError
        # The Ruby process is already exiting; a failing Python handler must not turn that into an error.
        nil
      end

      def initialize
        @mutex = Mutex.new
      end

      def train_lora(config)
        call(:train_lora, config, default_error: TrainingError)
      end

      def merge_lora(config)
        call(:merge_lora, config, default_error: MergeError)
      end

      def generate(config)
        call(:generate, config, default_error: GenerationError)
      end

      def detect_device(config = {})
        call(:detect_device, config, default_error: DependencyError)
      end

      def diagnostics
        call(:diagnostics, nil, default_error: DependencyError)
      end

      private

      def call(name, config, default_error:)
        facade = backend_module
        result = if config
          facade.public_send(name, JSON.generate(config))
        else
          facade.public_send(name)
        end

        JSON.parse(result.to_s)
      rescue Error
        raise
      rescue StandardError => e
        # Raising inside the rescue sets #cause to the original error,
        # so the Python traceback stays reachable for debugging.
        raise ErrorTranslation.translate(e, default: default_error)
      end

      def backend_module
        @mutex.synchronize do
          @backend_module ||= import_backend
        end
      end

      def import_backend
        # Both must be exported before the embedded Python imports torch/tokenizers,
        # so they are set here rather than in Python.
        ENV['TOKENIZERS_PARALLELISM'] ||= 'false'
        ENV['PYTORCH_ENABLE_MPS_FALLBACK'] ||= '1'
        python = self.class.default_python
        ENV['PYTHON'] = python if python

        configure_venv

        require_pycall

        import_backend_module
      end

      # PYTHONPATH is read at interpreter initialization, so this must run
      # before PyCall first touches Python.
      def configure_venv
        site_packages = self.class.venv_site_packages(ENV.fetch('PYTHON', nil))
        return unless site_packages

        entries = ENV['PYTHONPATH'].to_s.split(File::PATH_SEPARATOR)
        return if entries.include?(site_packages)

        ENV['PYTHONPATH'] = [site_packages, *entries].join(File::PATH_SEPARATOR)
      end

      def require_pycall
        require 'pycall'
      rescue LoadError
        raise DependencyError, 'PyCall is not installed. Add `gem "pycall"` to your Gemfile and run `bundle install`.'
      end

      # Referencing PyCall constants is only safe once require_pycall has succeeded,
      # hence the separate method.
      def import_backend_module
        # Unconditional insert: this runs once per process (memoized in backend_module),
        # and checking membership via sys.path.to_a triggers PyCall list-conversion warnings.
        sys = PyCall.import_module('sys')
        sys.path.insert(0, PYTHON_DIR)

        backend = PyCall.import_module('grain_backend')
        install_exit_hook(PyCall.import_module('atexit'))
        backend
      rescue PyCall::PyError => e
        raise DependencyError, "failed to import the Grain Python backend: #{e.message}"
      rescue PyCall::Error => e
        raise DependencyError, "Python interpreter not found or PyCall failed to initialize (#{e.message}). " \
                               'PyCall needs a Python with a shared libpython built for the same CPU ' \
                               "architecture as Ruby; static builds (pyenv's default) and an x86_64 Ruby " \
                               'under Rosetta with an arm64 Python do not work. Set the PYTHON environment ' \
                               'variable to a compatible interpreter, e.g. PYTHON=$(command -v python3.12).'
      end

      def install_exit_hook(atexit)
        at_exit { self.class.run_exit_handlers(atexit) }
      end
    end
  end
end
