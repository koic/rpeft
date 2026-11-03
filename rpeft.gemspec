# frozen_string_literal: true

require_relative 'lib/rpeft/version'

Gem::Specification.new do |spec|
  spec.name = 'rpeft'
  spec.version = RPEFT::VERSION
  spec.authors = ['Koichi ITO']
  spec.email = ['koic.ito@gmail.com']

  spec.summary = 'LoRA fine-tuning for LLMs from Ruby.'
  spec.description = 'A Ruby interface for LoRA fine-tuning of large language models. ' \
                     'It drives PyTorch, Hugging Face Transformers, and PEFT through PyCall, ' \
                     'modeling the train, adapter, and merge lifecycle with a Ruby API and CLI.'
  spec.homepage = 'https://github.com/koic/rpeft'
  spec.license = 'MIT'
  spec.required_ruby_version = '>= 3.0.0'

  spec.metadata['allowed_push_host'] = 'https://rubygems.org'

  spec.metadata['homepage_uri'] = spec.homepage
  spec.metadata['changelog_uri'] = 'https://github.com/koic/rpeft/blob/master/CHANGELOG.md'
  spec.metadata['rubygems_mfa_required'] = 'true'

  spec.files = Dir.glob(%w(lib/**/*.rb python/**/*.py exe/*), base: __dir__)
  spec.extra_rdoc_files = ['LICENSE.txt', 'README.md']
  spec.bindir = 'exe'
  spec.executables = spec.files.grep(%r|\Aexe/|) { |f| File.basename(f) }
  spec.require_paths = ['lib']

  # PyCall requires fiddle, which is no longer a default gem since Ruby 4.0.
  spec.add_dependency 'fiddle'
  spec.add_dependency 'pycall', '~> 1.5'
end
