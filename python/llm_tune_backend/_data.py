from . import _chat
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
                texts = [_chat.render(tokenizer, messages) for messages in batch[field]]
            else:
                texts = batch[field]

            # No padding here: the collator pads each batch dynamically.
            return tokenizer(texts, truncation=True, max_length=max_length)

        return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
