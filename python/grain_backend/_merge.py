import gc

from . import _models
from ._errors import stage


def merge_lora(config):
    with stage("merge"):
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM

        adapter_path = config["adapter_path"]
        output_dir = config["output_dir"]
        base_model = config.get("base_model") or PeftConfig.from_pretrained(
            adapter_path
        ).base_model_name_or_path

        merged = None
        try:
            # Merging is elementwise weight math; CPU is
            # the memory-safest place for it and speed is irrelevant.
            base = AutoModelForCausalLM.from_pretrained(base_model, dtype="auto")
            model = PeftModel.from_pretrained(base, adapter_path)
            merged = model.merge_and_unload()
            merged.save_pretrained(output_dir, safe_serialization=True)

            # From the base model, the same tokenizer generation with the adapter uses. The adapter
            # directory may carry a tokenizer and chat template of its own, which would otherwise
            # make the merged model behave differently from the adapter it was merged from.
            tokenizer = _models.load_tokenizer(base_model)
            tokenizer.save_pretrained(output_dir)

            dtype = str(next(merged.parameters()).dtype).replace("torch.", "")
            return {"model_path": output_dir, "base_model": base_model, "dtype": dtype}
        finally:
            merged = None
            gc.collect()
