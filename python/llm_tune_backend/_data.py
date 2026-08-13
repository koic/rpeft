from ._errors import BackendError, stage


def load_tokenized_dataset(path, tokenizer, text_field, max_length):
    with stage("dataset"):
        from datasets import load_dataset

        dataset = load_dataset("json", data_files=path, split="train")
        if text_field not in dataset.column_names:
            raise BackendError(
                "dataset",
                "field '%s' not found in dataset (available: %s)"
                % (text_field, ", ".join(dataset.column_names)),
            )

        def tokenize(batch):
            # No padding here: the collator pads each batch dynamically.
            return tokenizer(batch[text_field], truncation=True, max_length=max_length)

        return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
