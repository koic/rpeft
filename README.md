# LLM Tune

LLM Tune is a Ruby gem for LoRA fine-tuning of large language models. It provides a Ruby API and CLI that model the LoRA lifecycle,
while the heavy lifting is done by the Python ecosystem (PyTorch, Hugging Face Transformers, PEFT, Datasets, Accelerate) through PyCall.

```text
Ruby application / CLI
        |
        v
     llm_tune
        |
        v
      PyCall
        |
        v
Python backend (bundled, private)
   |- transformers
   |- peft
   |- datasets
   `- torch
        |
        v
CUDA / MPS / CPU
```

llm_tune does not reimplement Transformers or LoRA in Ruby, and it does not hide what fine-tuning is. It hides Python plumbing, not concepts.

## Status

Early proof of concept (v0.1). The lifecycle model of the public API is expected to stay, but details may change until v1.0.

## The model behind the API

LoRA keeps the base model weights `W` frozen and learns a low-rank update of rank `r`:

```text
W' = W + delta W
delta W ~= B A
```

`A` and `B` are small matrices. Training produces them; merging adds `B A` back into `W`. The API is the same shape:

```text
Base Model
    |
    | train
    v
LoRA Adapter
    |
    | merge
    v
Merged Model
```

The `rank:` option in the API is exactly the `r` above.

## Requirements

- Ruby 3.0+
- Python 3.9+ with a shared libpython (static Python builds such as pyenv's default do not work; tested with Python 3.12)
- Ruby and Python built for the same CPU architecture. On Apple Silicon, an x86_64 Ruby running under Rosetta cannot load an arm64 libpython;
  a symbol error such as `_Py_NoneStruct` from `llm-tune doctor` usually means this mismatch
- The Python packages `torch`, `transformers`, `peft`, `datasets`, and `accelerate`

## Installation

Add the gem to your Gemfile:

```console
$ bundle add llm_tune
```

Set up a Python environment. With venv:

```console
$ python3 -m venv .venv
$ .venv/bin/pip install torch transformers peft datasets accelerate
```

Or with uv:

```console
$ uv venv
$ uv pip install torch transformers peft datasets accelerate
```

Note for pyenv users: PyCall needs a shared libpython, so install the interpreter with:

```console
$ PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.12
```

When a `.venv` directory exists in the current directory, llm_tune uses it automatically. Set the `PYTHON` environment variable to override the interpreter,
then check the environment:

```console
$ bundle exec llm-tune doctor
```

`llm-tune doctor` reports the Python interpreter, each required package, and the available devices, and tells you what is missing.

## Usage

```ruby
require 'llm_tune'

adapter = LLMTune::LoRA.train(
  model: 'Qwen/Qwen2.5-Coder-0.5B',
  dataset: 'ruby_answers.jsonl',
  rank: 8, # The LoRA rank r
  alpha: 16,
  dropout: 0.05,
  epochs: 2
)
adapter.save('./ruby-lora')

model = adapter.merge # W' = W + B A
model.save('./ruby-model')
```

`LLMTune::LoRA.train` returns an `LLMTune::LoRA::Adapter` that knows its `path`, `base_model`, `rank`, `alpha`, `dropout`, `target_modules`, and training `metadata`.
`adapter.merge` returns an `LLMTune::Model` with `path` and `base_model`. `save` moves the artifact to the given path; a destination produced by a previous `save` is replaced,
so scripts can be re-run, while any other existing path is refused.

### Generation

`LLMTune::Model#generate` produces a completion for a prompt. It uses greedy decoding, so the output is deterministic for a given model and prompt.
The model can be a merged model directory or a Hugging Face model id, which makes a before and after comparison one call each:

```ruby
prompt = "User: How do I reverse an array?\nAssistant:"

base = LLMTune::Model.new(path: 'Qwen/Qwen2.5-Coder-0.5B')
base.generate(prompt, max_new_tokens: 64)  # English prose from the base model

model = adapter.merge
model.generate(prompt, max_new_tokens: 64) # Ruby code from the fine-tuned model
```

Passing `temperature:` or `top_p:` switches decoding from greedy to sampling, which makes the model produce
different wording on each call. `seed:` makes sampled output reproducible again:

```ruby
model.generate(prompt, temperature: 0.7, top_p: 0.9, seed: 42)
```

### Options

| Option            | Default | Meaning                                                                                 |
| ----------------- | ------- | --------------------------------------------------------------------------------------- |
| `rank:`           | `8`     | LoRA rank `r`, the rank of the low-rank update `B A`                                    |
| `alpha:`          | `16`    | LoRA scaling factor (`lora_alpha` in PEFT)                                              |
| `dropout:`        | `0.05`  | LoRA dropout (`lora_dropout` in PEFT)                                                   |
| `target_modules:` | `nil`   | Modules to adapt, e.g. `%w[q_proj v_proj]`. `nil` uses PEFT's per-architecture defaults |
| `epochs:`         | `3`     | Training epochs                                                                         |
| `batch_size:`     | `4`     | Per-device batch size                                                                   |
| `learning_rate:`  | `2e-4`  | Learning rate                                                                           |
| `max_length:`     | `512`   | Max token length per example                                                            |
| `output_dir:`     | `nil`   | Adapter output directory. `nil` uses a temporary directory; `save` moves it             |
| `device:`         | `:auto` | `:auto`, `:cuda`, `:mps`, or `:cpu`                                                     |

### Dataset

v0.1 supports JSONL files where each line is a JSON object with a `text` key:

```json
{"text":"User: How do I reverse an array?\nAssistant: [1, 2, 3].reverse\n# => [3, 2, 1]"}
{"text":"User: How do I get the length of a string?\nAssistant: 'hello'.length\n# => 5"}
```

A useful fine-tuning dataset teaches the model something it does not already do. The bundled `examples/ruby_answers.jsonl` teaches the model to answer programming questions with Ruby code only:
the base model replies with an English paragraph, while the fine-tuned model replies with a bare snippet and a `# =>` comment. The effect is visible at a glance in any spoken language,
because the difference is the shape of the output rather than its wording. See `examples/train_and_merge.rb` for a runnable example.

### Devices and Apple Silicon

The device is auto-detected with CUDA > MPS > CPU priority:

```ruby
LLMTune.device # => :mps
```

MacBook Pro (Apple Silicon, M1 and later) is a first-class target: small models such as `Qwen/Qwen2.5-Coder-0.5B` fine-tune on MPS.
Training uses float32 on MPS (half precision there is numerically fragile), bfloat16 on capable CUDA GPUs, and `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically for
operators MPS does not implement. Merging always runs on CPU, which is the memory-safest place for it.

### Environment check

```ruby
LLMTune.check # => a Hash describing Python, packages, and devices
```

```console
$ llm-tune doctor
Ruby:         3.3.4 (llm_tune 0.1.0)
Python:       3.12.4 (/path/to/.venv/bin/python)
torch:        OK 2.4.0
transformers: OK 4.44.0
peft:         OK 0.12.0
datasets:     OK 2.21.0
accelerate:   OK 0.33.0
Device:       mps (CUDA: no, MPS: yes)
Status:       OK
```

### Errors

Python exceptions never surface raw. They are translated into `LLMTune::DependencyError`, `LLMTune::ModelLoadError`, `LLMTune::DatasetError`, `LLMTune::TrainingError`,
or `LLMTune::MergeError`, and the original Python error (with its traceback) stays reachable via `error.cause`.

## CLI

The CLI mirrors the Ruby API:

```console
$ llm-tune lora train \
    --model Qwen/Qwen2.5-Coder-0.5B \
    --dataset ruby_answers.jsonl \
    --rank 8 \
    --epochs 2 \
    --output ruby-lora
```

```console
$ llm-tune lora merge \
    --adapter ruby-lora \
    --output ruby-model
```

`lora merge` reads the base model from the metadata recorded in the adapter directory; pass `--model` to override it (for example, for adapters produced by plain PEFT).

```console
$ llm-tune generate \
    --model ruby-model \
    --prompt $'User: How do I reverse an array?\nAssistant:'
```

`--model` also accepts a Hugging Face model id, so the same command shows the base model's output for comparison.
`--temperature`, `--top-p`, and `--seed` are the CLI equivalents of the sampling options above.
`llm-tune doctor` and `llm-tune version` are also available.

## Process model and limitations

llm_tune is designed for CLI, Rake tasks, batch jobs, and offline experiments. It is not designed to run inside web application workers:

- PyCall embeds a single Python interpreter in the Ruby process. Backend calls hold the Python GIL and block the calling Ruby thread for the whole training run.
- Do not fork the process after the backend has been initialized (Spring, Puma workers, and so on). The backend loads lazily, so forking before the first call is safe.
- Training keeps no checkpoints in v0.1 (`save_strategy: "no"`), so a crash during training loses that run's progress.

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then run `bundle exec rake` for the unit tests and RuboCop. Unit tests never load PyCall or Python;
the Python backend is replaced by a fake at the backend seam (`LLMTune.backend=`).

An integration test runs a real fine-tune with a tiny model on CPU. It needs a prepared Python environment and network access for the first model download:

```console
$ LLM_TUNE_INTEGRATION=1 bundle exec rake test:integration
```

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/koic/llm_tune.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).
