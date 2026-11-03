import gc
import os
import shutil

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
            dataset_format=config.get("dataset_format", "text"),
        )
        eval_dataset = None
        if config.get("eval_dataset_path"):
            eval_dataset = _data.load_tokenized_dataset(
                config["eval_dataset_path"],
                tokenizer,
                text_field=config.get("text_field", "text"),
                max_length=config.get("max_length", 512),
                dataset_format=config.get("dataset_format", "text"),
            )

        with stage("training"):
            model, target_modules, param_counts = _apply_lora(model, lora)
            trainer = _build_trainer(model, tokenizer, dataset, eval_dataset, config, device, dtype)
            resume = _resume_point(config)
            metrics = trainer.train(resume_from_checkpoint=resume).metrics
            steps = trainer.state.global_step
            if eval_dataset is not None:
                metrics = dict(metrics, **trainer.evaluate())

        with stage("adapter_save"):
            model.save_pretrained(output_dir)
            # Checkpoints only serve resuming an interrupted run. Once the adapter is saved they would
            # just travel with it wherever the adapter directory is saved or shared.
            shutil.rmtree(_trainer_dir(output_dir), ignore_errors=True)

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


def _build_trainer(model, tokenizer, dataset, eval_dataset, config, device, dtype):
    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(
        output_dir=_trainer_dir(config["output_dir"]),
        num_train_epochs=config.get("epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        learning_rate=config.get("learning_rate", 2e-4),
        logging_steps=config.get("logging_steps", 5),
        logging_strategy="steps",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy=config.get("save_strategy", "no"),
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
        eval_dataset=eval_dataset,
        data_collator=_collator(tokenizer),
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
        "eval_loss": float(metrics["eval_loss"]) if "eval_loss" in metrics else None,
        "num_examples": int(num_examples),
        "steps": int(steps),
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "effective_config": effective,
    }


def _trainer_dir(output_dir):
    return os.path.join(output_dir, ".trainer")


def _resume_point(config):
    if not config.get("resume_from_checkpoint"):
        return None

    from transformers.trainer_utils import get_last_checkpoint

    # A finished run removes its checkpoints, so there may be nothing to resume; train from the start then.
    trainer_dir = _trainer_dir(config["output_dir"])
    return get_last_checkpoint(trainer_dir) if os.path.isdir(trainer_dir) else None


def _collator(tokenizer):
    # DataCollatorForLanguageModeling ignores every position holding the pad token. Tokenizers without
    # a pad token of their own pad with EOS (see _models.load_tokenizer), and many that have one reuse
    # EOS too, so the EOS that ends each example would never be learned. Only the padding the batch
    # added is ignored here, found through the attention mask instead of the token id.
    def collate(features):
        batch = tokenizer.pad(features, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

    return collate
