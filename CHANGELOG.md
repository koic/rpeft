# Changelog

## 0.1.0 (unreleased)

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
