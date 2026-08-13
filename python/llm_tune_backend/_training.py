import gc
import os

from . import _data, _device, _models
from ._errors import stage


def train_lora(config):
    lora = config.get("lora", {})
    resolved = _device.resolve_device(config.get("device"))
    device = resolved["device"]
    dtype = resolved["dtype"]
    output_dir = config["output_dir"]

    model = None
    try:
        tokenizer, model = _models.load_tokenizer_and_model(
            config["base_model"],
            dtype=dtype,
            device=device,
            trust_remote_code=config.get("trust_remote_code", False),
        )
        dataset = _data.load_tokenized_dataset(
            config["dataset_path"],
            tokenizer,
            text_field=config.get("text_field", "text"),
            max_length=config.get("max_length", 512),
        )

        with stage("training"):
            model, target_modules, param_counts = _apply_lora(model, lora)
            trainer = _build_trainer(model, tokenizer, dataset, config, device, dtype)
            metrics = trainer.train().metrics
            steps = trainer.state.global_step

        with stage("adapter_save"):
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        return _result(
            config, device, dtype, target_modules, param_counts, metrics, steps, len(dataset)
        )
    finally:
        model = None
        gc.collect()
        _device.empty_cache(device)


def _apply_lora(model, lora):
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=lora.get("r", 8),
        lora_alpha=lora.get("alpha", 16),
        lora_dropout=lora.get("dropout", 0.05),
        # None lets PEFT pick per-architecture defaults (e.g. q_proj/v_proj).
        target_modules=lora.get("target_modules"),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    target_modules = model.peft_config["default"].target_modules
    if isinstance(target_modules, (set, frozenset)):
        target_modules = sorted(target_modules)
    elif isinstance(target_modules, str):
        target_modules = [target_modules]
    else:
        target_modules = list(target_modules)

    return model, target_modules, model.get_nb_trainable_parameters()


def _build_trainer(model, tokenizer, dataset, config, device, dtype):
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    args = TrainingArguments(
        output_dir=os.path.join(config["output_dir"], ".trainer"),
        num_train_epochs=config.get("epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        learning_rate=config.get("learning_rate", 2e-4),
        logging_steps=config.get("logging_steps", 5),
        logging_strategy="steps",
        save_strategy="no",
        report_to=[],
        seed=config.get("seed", 42),
        bf16=(dtype == "bfloat16"),
        fp16=False,
        dataloader_pin_memory=(device == "cuda"),
        use_cpu=(device == "cpu"),
        optim="adamw_torch",
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )


def _result(config, device, dtype, target_modules, param_counts, metrics, steps, num_examples):
    trainable_params, total_params = param_counts
    effective = dict(config)
    effective["device"] = device
    effective["lora"] = dict(config.get("lora", {}))
    effective["lora"]["target_modules"] = target_modules

    return {
        "adapter_path": config["output_dir"],
        "base_model": config["base_model"],
        "device": device,
        "dtype": dtype,
        "train_runtime_seconds": float(metrics.get("train_runtime", 0.0)),
        "final_loss": float(metrics.get("train_loss", 0.0)),
        "num_examples": int(num_examples),
        "steps": int(steps),
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "effective_config": effective,
    }
