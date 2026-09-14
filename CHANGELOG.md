# Changelog

## 0.1.0 (unreleased)

- LoRA fine-tuning from Ruby (`Grain::LoRA.train`) via PyCall, PyTorch, Hugging Face Transformers, and PEFT
- `Grain::LoRA::Adapter` and `Grain::Model` objects modeling the train, adapter, and merge lifecycle
- Adapter merging (`adapter.merge`) using PEFT's `merge_and_unload`
- Text generation with greedy decoding from a base or fine-tuned model (`Grain::Model#generate`, `grain generate`)
- Sampling options for text generation (`temperature:`, `top_p:`, `seed:`)
- Adapter inference without merging (`Grain::LoRA::Adapter#generate`, `grain generate --adapter`)
- Instruction-format datasets (`instruction`/`input`/`output` JSONL lines)
- Chat-format datasets (`messages` JSONL lines) rendered with the tokenizer's chat template
- Chat inference (`Grain::Model#chat`, `Grain::LoRA::Adapter#chat`) applying the model's chat template
- Enumerable datasets: `dataset:` accepts record Hashes built in Ruby, with no intermediate file
- Evaluation during training (`eval_dataset:`), with the final `eval_loss` recorded in the adapter metadata
- Per-epoch checkpoints (`checkpoints:`) and resuming an interrupted run (`resume:`)
- Device auto-detection with CUDA > MPS > CPU priority (`Grain.device`)
- JSONL dataset support (`{"text": "..."}` lines)
- `grain` CLI: `lora train`, `lora merge`, `doctor`, `version`
- Environment diagnostics (`Grain.check` / `grain doctor`)
