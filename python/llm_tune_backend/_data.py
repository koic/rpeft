from ._errors import BackendError, stage


def load_tokenized_dataset(path, tokenizer, text_field, max_length, dataset_format="text"):
    with stage("dataset"):
        from datasets import load_dataset

        dataset = load_dataset("json", data_files=path, split="train")
        field = "messages" if dataset_format == "messages" else text_field
        if field not in dataset.column_names:
            raise BackendError(
                "dataset",
                "field '%s' not found in dataset (available: %s)"
                % (field, ", ".join(dataset.column_names)),
            )

        def tokenize(batch):
            if dataset_format == "messages":
                texts = [_render_chat(tokenizer, messages) for messages in batch[field]]
            else:
                texts = batch[field]

            # No padding here: the collator pads each batch dynamically.
            return tokenizer(texts, truncation=True, max_length=max_length)

        return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


def _render_chat(tokenizer, messages):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # Tokenizers without a chat template (small test models, some base models) still need
    # a deterministic rendering.
    return "\n".join("%s: %s" % (message["role"], message["content"]) for message in messages)
