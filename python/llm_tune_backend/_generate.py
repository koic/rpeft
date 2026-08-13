import gc

from . import _device, _models
from ._errors import stage


def generate(config):
    resolved = _device.resolve_device(config.get("device"))
    device = resolved["device"]
    dtype = resolved["dtype"]

    model = None
    try:
        tokenizer, model = _models.load_tokenizer_and_model(
            config["model"],
            dtype=dtype,
            device=device,
            trust_remote_code=config.get("trust_remote_code", False),
        )

        adapter_path = config.get("adapter_path")
        if adapter_path:
            with stage("model_load"):
                from peft import PeftModel

                # Applied on top of the base weights; not merged.
                model = PeftModel.from_pretrained(model, adapter_path)

        with stage("generation"):
            import torch

            if config.get("seed") is not None:
                torch.manual_seed(config["seed"])

            # Greedy decoding by default keeps the output deterministic
            # for a given model and prompt, which demo scripts rely on;
            # temperature or top_p switches sampling on.
            generation_options = {
                "max_new_tokens": config.get("max_new_tokens", 64),
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if config.get("temperature") is not None:
                generation_options["temperature"] = float(config["temperature"])
                generation_options["do_sample"] = True
            if config.get("top_p") is not None:
                generation_options["top_p"] = float(config["top_p"])
                generation_options["do_sample"] = True

            inputs = tokenizer(config["prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation_options)
            completion_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

        return {
            "completion": str(completion),
            "model": config["model"],
            "device": device,
            "new_tokens": int(completion_ids.shape[0]),
        }
    finally:
        model = None
        gc.collect()
        _device.empty_cache(device)
