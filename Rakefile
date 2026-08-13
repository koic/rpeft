# frozen_string_literal: true

require 'bundler/gem_tasks'
require 'minitest/test_task'
require 'rubocop/rake_task'

Minitest::TestTask.create(:test) do |t|
  t.test_globs = ['test/test_*.rb', 'test/lora/**/test_*.rb', 'test/cli/**/test_*.rb']
end

namespace :test do
  # Requires a Python environment; see test/integration/.
  Minitest::TestTask.create(:integration) do |t|
    t.test_globs = ['test/integration/**/test_*.rb']
  end
end

RuboCop::RakeTask.new

task default: %i(test rubocop)
