# frozen_string_literal: true

require_relative 'lib/grain/version'

Gem::Specification.new do |spec|
  spec.name = 'grain'
  spec.version = Grain::VERSION
  spec.authors = ['Koichi ITO']
  spec.email = ['koic.ito@gmail.com']

  spec.summary = 'LoRA fine-tuning for LLMs from Ruby.'
  spec.description = 'A Ruby interface for LoRA fine-tuning of large language models. ' \
                     'It drives PyTorch, Hugging Face Transformers, and PEFT through PyCall, ' \
                     'modeling the train, adapter, and merge lifecycle with a Ruby API and CLI.'
  spec.homepage = 'https://github.com/koic/grain'
  spec.license = 'MIT'
  spec.required_ruby_version = '>= 3.0.0'

  spec.metadata['allowed_push_host'] = 'https://rubygems.org'

  spec.metadata['homepage_uri'] = spec.homepage
  spec.metadata['source_code_uri'] = 'https://github.com/koic/grain'
  spec.metadata['changelog_uri'] = 'https://github.com/koic/grain/blob/master/CHANGELOG.md'
  spec.metadata['rubygems_mfa_required'] = 'true'

  # Specify which files should be added to the gem when it is released.
  # The `git ls-files -z` loads the files in the RubyGem that have been added into git.
  # Note: the bundled python/ backend ships with the gem via this list.
  gemspec = File.basename(__FILE__)
  spec.files = IO.popen(%w(git ls-files -z), chdir: __dir__, err: IO::NULL) { |ls|
    ls.readlines("\x0", chomp: true).reject do |f|
      (f == gemspec) ||
        f.start_with?(*%w(bin/ test/ spec/ features/ .git .rubocop appveyor Gemfile examples/))
    end
  }
  spec.bindir = 'exe'
  spec.executables = spec.files.grep(%r|\Aexe/|) { |f| File.basename(f) }
  spec.require_paths = ['lib']

  # PyCall requires fiddle, which is no longer a default gem since Ruby 4.0.
  spec.add_dependency 'fiddle'
  spec.add_dependency 'pycall', '~> 1.5'
end
