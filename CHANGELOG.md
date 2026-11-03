# Changelog

## 0.1.0 (unreleased)

- LoRA fine-tuning from Ruby (`RPEFT::LoRA.train`) via PyCall, PyTorch, Hugging Face Transformers, and PEFT
- `RPEFT::LoRA::Adapter` and `RPEFT::Model` objects modeling the train, adapter, and merge lifecycle
- Adapter merging (`adapter.merge`) using PEFT's `merge_and_unload`
- Text generation with greedy decoding from a base or fine-tuned model (`RPEFT::Model#generate`, `rpeft generate`)
- Sampling options for text generation (`temperature:`, `top_p:`, `seed:`)
- Adapter inference without merging (`RPEFT::LoRA::Adapter#generate`, `rpeft generate --adapter`)
- Instruction-format datasets (`instruction`/`input`/`output` JSONL lines)
- Chat-format datasets (`messages` JSONL lines) rendered with the tokenizer's chat template
- Chat inference (`RPEFT::Model#chat`, `RPEFT::LoRA::Adapter#chat`) applying the model's chat template
- Enumerable datasets: `dataset:` accepts record Hashes built in Ruby, with no intermediate file
- Evaluation during training (`eval_dataset:`), with the final `eval_loss` recorded in the adapter metadata
- Per-epoch checkpoints (`checkpoints:`) and resuming an interrupted run (`resume:`)
- Device auto-detection with CUDA > MPS > CPU priority (`RPEFT.device`)
- JSONL dataset support (`{"text": "..."}` lines)
- `rpeft` CLI: `lora train`, `lora merge`, `doctor`, `version`
- Environment diagnostics (`RPEFT.check` / `rpeft doctor`)
