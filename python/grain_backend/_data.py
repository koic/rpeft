import json

from . import _chat
from ._errors import BackendError, stage


def load_tokenized_dataset(path, tokenizer, text_field, max_length, dataset_format="text"):
    with stage("dataset"):
        from datasets import Dataset

        # Built in memory rather than with load_dataset("json", ...), which writes an Arrow copy of every
        # file into the Hugging Face cache. Each run reads a fresh file, so that copy would never be reused.
        dataset = Dataset.from_list(_read_jsonl(path))
        field = "messages" if dataset_format == "messages" else text_field
        if field not in dataset.column_names:
            raise BackendError(
                "dataset",
                "field '%s' not found in dataset (available: %s)"
                % (field, ", ".join(dataset.column_names)),
            )

        add_special_tokens = not (dataset_format == "messages" and _chat.writes_special_tokens(tokenizer))

        def tokenize(batch):
            if dataset_format == "messages":
                texts = [_chat.render(tokenizer, messages) for messages in batch[field]]
            else:
                texts = batch[field]

            # No padding here: the collator pads each batch dynamically.
            return tokenizer(
                texts, truncation=True, max_length=max_length, add_special_tokens=add_special_tokens
            )

        return dataset.map(
            tokenize, batched=True, remove_columns=dataset.column_names, keep_in_memory=True
        )


def _read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as lines:
        for lineno, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise BackendError("dataset", "line %d is not valid JSON: %s" % (lineno, error))
    return records
