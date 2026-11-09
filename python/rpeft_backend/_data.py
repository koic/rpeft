import json
import sys

from . import _chat
from ._errors import BackendError, stage

IGNORED = -100

# The record fields each backend format reads. text_field stands in for the configurable text column.
FIELDS = {"text": ("text_field",), "completion": ("prompt", "completion"), "messages": ("messages",)}


def load_tokenized_dataset(
    path, tokenizer, text_field, max_length, dataset_format="text", train_on_prompt=False, name="dataset"
):
    """Returns the tokenized dataset and the number of examples left out of it.

    Every example carries its own labels: the prompt positions hold IGNORED unless train_on_prompt is set,
    so the loss covers only what the model is meant to write. name tells the datasets apart in messages.
    """
    with stage("dataset"):
        from datasets import Dataset

        # Built in memory rather than with load_dataset("json", ...), which writes an Arrow copy of every
        # file into the Hugging Face cache. Each run reads a fresh file, so that copy would never be reused.
        records = _read_jsonl(path)
        fields = [text_field if field == "text_field" else field for field in FIELDS[dataset_format]]
        _check_fields(records, fields)

        add_special_tokens = not (dataset_format == "messages" and _chat.writes_special_tokens(tokenizer))
        # A chat template closes every turn with its own end marker. Plain text has none, so without an EOS
        # the model never sees where an answer ends and keeps generating until max_new_tokens.
        eos = tokenizer.eos_token_id if dataset_format != "messages" else None
        content_length = max_length - 1 if eos is not None else max_length

        examples = []
        for record in records:
            segments = _segments(tokenizer, record, dataset_format, fields, train_on_prompt)
            example = _encode(tokenizer, segments, add_special_tokens, content_length, train_on_prompt, eos)
            if example is not None:
                examples.append(example)

        skipped = len(records) - len(examples)
        reason = (
            "no answer left to train on: a transcript without an assistant reply after its first message, "
            "or an answer cut off by max_length %d" % max_length
        )
        if not examples:
            raise BackendError("dataset", "every example in %s has %s" % (name, reason))
        if skipped:
            print(
                "rpeft: skipped %d of %d examples in %s with %s" % (skipped, len(records), name, reason),
                file=sys.stderr,
            )

        return Dataset.from_list(examples), skipped


def _check_fields(records, fields):
    available = set()
    for record in records:
        available.update(record)
    for field in fields:
        if field not in available:
            raise BackendError(
                "dataset",
                "field '%s' not found in dataset (available: %s)" % (field, ", ".join(sorted(available))),
            )


def _segments(tokenizer, record, dataset_format, fields, train_on_prompt):
    # (text, trainable) pairs whose concatenation is the whole example.
    if dataset_format == "messages":
        if train_on_prompt:
            return [(_chat.render(tokenizer, record[fields[0]]), True)]
        return _chat.segments(tokenizer, record[fields[0]])
    if dataset_format == "completion":
        return [(record[fields[0]], False), (record[fields[1]], True)]
    return [(record[fields[0]], True)]


def _encode(tokenizer, segments, add_special_tokens, content_length, train_on_prompt, eos):
    def tokenize(text):
        return tokenizer(
            text, truncation=True, max_length=content_length, add_special_tokens=add_special_tokens
        )["input_ids"]

    # The example is tokenized as a whole, so every token is the one the model sees in the full text.
    # Tokenizing segments one by one would not do: a SentencePiece tokenizer marks the start of each piece
    # with a space, turning an answer "hello" into " hello". A segment ends where the text up to its end,
    # tokenized on its own as the prompt is at inference, stops matching the whole.
    text = "".join(segment for segment, _ in segments)
    input_ids = tokenize(text)
    labels = []
    answers = []
    start = 0
    offset = 0
    for segment, trainable in segments:
        offset += len(segment)
        end = len(input_ids) if offset == len(text) else max(start, _common_prefix(tokenize(text[:offset]), input_ids))
        ids = input_ids[start:end]
        labels.extend(ids if trainable or train_on_prompt else [IGNORED] * len(ids))
        answers.extend([trainable] * len(ids))
        start = end

    if eos is not None:
        input_ids.append(eos)
        labels.append(eos)

    # Without prompt training, an answer cut away entirely leaves an example that teaches nothing but to stop.
    answer_cut_away = any(trainable and segment for segment, trainable in segments) and not any(answers)
    if (not train_on_prompt and answer_cut_away) or all(label == IGNORED for label in labels):
        return None

    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def _common_prefix(ids, other):
    length = 0
    for left, right in zip(ids, other):
        if left != right:
            break
        length += 1
    return length


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
