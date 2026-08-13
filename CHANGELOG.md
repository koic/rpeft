# Changelog

## 0.1.0 (unreleased)

- LoRA fine-tuning from Ruby (`LLMTune::LoRA.train`) via PyCall, PyTorch, Hugging Face Transformers, and PEFT
- `LLMTune::LoRA::Adapter` and `LLMTune::Model` objects modeling the train, adapter, and merge lifecycle
- Adapter merging (`adapter.merge`) using PEFT's `merge_and_unload`
- Text generation with greedy decoding from a base or fine-tuned model (`LLMTune::Model#generate`, `llm-tune generate`)
- Sampling options for text generation (`temperature:`, `top_p:`, `seed:`)
- Adapter inference without merging (`LLMTune::LoRA::Adapter#generate`, `llm-tune generate --adapter`)
- Instruction-format datasets (`instruction`/`input`/`output` JSONL lines)
- Chat-format datasets (`messages` JSONL lines) rendered with the tokenizer's chat template
- Device auto-detection with CUDA > MPS > CPU priority (`LLMTune.device`)
- JSONL dataset support (`{"text": "..."}` lines)
- `llm-tune` CLI: `lora train`, `lora merge`, `doctor`, `version`
- Environment diagnostics (`LLMTune.check` / `llm-tune doctor`)
