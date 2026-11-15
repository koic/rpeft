# Changelog

## Unreleased

- A block given to `RPEFT::LoRA.train` receives the progress of the run (`RPEFT::LoRA::Progress`) at every logged step and evaluation.
  An exception raised in the block ends the run
- The logged steps are recorded in the adapter metadata (`training.history`), with or without a block
- Ctrl-C now ends a training run at the next logged step instead of after the last one

## 0.3.0 (2026-11-15)

A fine-tuned model can now leave RPEFT: exported to GGUF, it runs in Ollama, llama.cpp, and other local runtimes, with no Python where it is used.
Existing code behaves as it did in 0.2.0.

- GGUF export of a merged model (`RPEFT::Model#export_gguf`, `rpeft export gguf`) through llama.cpp's own tools,
  with Q8_0 by default and the K-quant types through `llama-quantize`. `chat_template: false` (`--no-chat-template`) leaves out
  the chat template for a model fine-tuned on plain text. `rpeft doctor` reports the llama.cpp tools it finds

## 0.2.0 (2026-11-10)

Upgrading changes what instruction and chat datasets train on, so the same data and options give a different adapter than in 0.1.0 (see the first entry).

- Instruction and chat datasets now train on the answers alone: the prompt is read as context but no longer counts toward the loss.
  `train_on_prompt: true` (`--train-on-prompt` in the CLI) restores training on every token. Text datasets are unchanged
- An assistant turn opening a chat transcript, such as a greeting, answers nothing and is now read as context rather than trained on
- Examples whose answer does not fit within `max_length:` are skipped with a warning and counted in the adapter metadata (`skipped_examples` and `eval_skipped_examples`)

## 0.1.0 (2026-11-04)

The first release. It brings LoRA fine-tuning of large language models to Ruby, from training an adapter
to merging it into a standalone model and generating text, on Apple Silicon, CUDA, or CPU.

- LoRA fine-tuning from Ruby (`RPEFT::LoRA.train`) through PyCall, PyTorch, Hugging Face Transformers, and PEFT
- `RPEFT::LoRA::Adapter` and `RPEFT::Model` objects modeling the train, adapter, and merge lifecycle
- JSONL datasets in plain text (`text`), instruction (`instruction`/`input`/`output`), and chat (`messages`) formats,
  or the same records as an Enumerable of Hashes built in Ruby
- Evaluation during training (`eval_dataset:`), with the final `eval_loss` recorded in the adapter metadata
- Per-epoch checkpoints (`checkpoints:`) and resuming an interrupted run (`resume:`)
- Adapter merging (`adapter.merge`) into a standalone model with PEFT's `merge_and_unload`
- Text generation (`RPEFT::Model#generate`), greedy by default, with sampling options (`temperature:`, `top_p:`, `seed:`)
- Chat inference (`RPEFT::Model#chat`) with the model's own chat template
- Generation and chat with an adapter applied on top of its base model, without merging
  (`RPEFT::LoRA::Adapter#generate`, `RPEFT::LoRA::Adapter#chat`)
- Device auto-detection with CUDA > MPS > CPU priority (`RPEFT.device`)
- Environment diagnostics (`RPEFT.check`) that report missing and outdated Python packages
- `rpeft` CLI: `lora train`, `lora merge`, `generate`, `doctor`, and `version`
- Requires torch 2.6 or later and transformers 4.56 or later
