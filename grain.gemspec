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

  spec.files = Dir['lib/**/*.rb', 'python/**/*.py', 'exe/*']
  spec.extra_rdoc_files = ['LICENSE.txt', 'README.md']
  spec.bindir = 'exe'
  spec.executables = spec.files.grep(%r|\Aexe/|) { |f| File.basename(f) }
  spec.require_paths = ['lib']

  # PyCall requires fiddle, which is no longer a default gem since Ruby 4.0.
  spec.add_dependency 'fiddle'
  spec.add_dependency 'pycall', '~> 1.5'
end
