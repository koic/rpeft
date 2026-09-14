from ._errors import stage


def load_tokenizer_and_model(base_model, dtype, device, trust_remote_code=False):
    with stage("model_load"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            base_model, use_fast=True, trust_remote_code=trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=getattr(torch, dtype),
            trust_remote_code=trust_remote_code,
        )
        model = model.to(device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        return tokenizer, model
