# frozen_string_literal: true

# Test double for the backend seam (see RPEFT::Backend). Records
# every call in #calls, returns canned responses that can be
# overridden per method via responses:, and raises error: when set.
class FakeBackend
  # datasets maps each dataset path passed to train_lora to its lines at call time, because RPEFT removes
  # its temporary copies once training returns.
  attr_reader :calls, :datasets

  def initialize(responses: {}, error: nil)
    @responses = responses
    @error = error
    @calls = []
    @datasets = {}
  end

  def self.healthy_diagnostics
    {
      'python' => {'executable' => '/usr/bin/python3', 'version' => '3.12.4'},
      'backend_version' => RPEFT::VERSION,
      'libraries' => {
        'torch' => {'installed' => true, 'version' => '2.6.0', 'minimum' => '2.6', 'outdated' => false},
        'transformers' => {'installed' => true, 'version' => '4.56.0', 'minimum' => '4.56', 'outdated' => false},
        'peft' => {'installed' => true, 'version' => '0.12.0'},
        'datasets' => {'installed' => true, 'version' => '2.21.0'},
        'accelerate' => {'installed' => true, 'version' => '0.33.0'}
      },
      'device' => {
        'cuda' => {'available' => false, 'device_name' => nil},
        'mps' => {'available' => true, 'built' => true},
        'recommended' => 'mps'
      }
    }
  end

  def train_lora(config)
    [config['dataset_path'], config['eval_dataset_path']].compact.each do |path|
      @datasets[path] = File.readlines(path) if File.file?(path)
    end
    record(:train_lora, config)
    @responses.fetch(:train_lora) do
      {
        'adapter_path' => config['output_dir'],
        'base_model' => config['base_model'],
        'device' => config['device'],
        'dtype' => 'float32',
        'train_runtime_seconds' => 1.5,
        'final_loss' => 2.5,
        'eval_loss' => config['eval_dataset_path'] ? 2.75 : nil,
        'num_examples' => 4,
        'skipped_examples' => 0,
        'eval_skipped_examples' => config['eval_dataset_path'] ? 0 : nil,
        'steps' => 4,
        'trainable_params' => 8192,
        'total_params' => 100_000,
        'effective_config' => effective_config(config)
      }
    end
  end

  def merge_lora(config)
    record(:merge_lora, config)
    @responses.fetch(:merge_lora) do
      {
        'model_path' => config['output_dir'],
        'base_model' => config['base_model'] || 'fake/base',
        'dtype' => 'float32'
      }
    end
  end

  def generate(config)
    record(:generate, config)
    @responses.fetch(:generate) do
      {
        'completion' => 'fake completion',
        'model' => config['model'],
        'device' => config['device'],
        'new_tokens' => 2
      }
    end
  end

  def detect_device(config = {})
    record(:detect_device, config)
    @responses.fetch(:detect_device) do
      {'device' => 'cpu', 'dtype' => 'float32', 'reason' => 'cpu fallback', 'device_name' => nil}
    end
  end

  def diagnostics
    record(:diagnostics, nil)
    @responses.fetch(:diagnostics) { self.class.healthy_diagnostics }
  end

  private

  def record(name, argument)
    raise @error if @error

    @calls << [name, argument]
  end

  def effective_config(config)
    effective = config.dup
    lora = (config['lora'] || {}).dup
    lora['target_modules'] ||= %w(q_proj v_proj)
    effective['lora'] = lora
    effective
  end
end
