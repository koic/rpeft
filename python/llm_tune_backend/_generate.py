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

        with stage("generation"):
            import torch

            inputs = tokenizer(config["prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                # Greedy decoding keeps the output deterministic for a
                # given model and prompt, which demo scripts rely on.
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=config.get("max_new_tokens", 64),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
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
