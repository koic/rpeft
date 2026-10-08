# Examples

`train_and_merge.rb` runs the whole LoRA lifecycle in one script: it asks the base model a question, trains a LoRA adapter on
`ruby_answers.jsonl`, merges the adapter into the base model, and asks the merged model the same question.

## Running

From the repository root, with the Python environment described in the [top-level README](../README.md#installation):

```console
$ bundle exec ruby examples/train_and_merge.rb
```

The first run downloads `Qwen/Qwen2.5-Coder-0.5B` from the Hugging Face Hub (about 1 GB). Training takes about 10 seconds on an M4 Max and longer on smaller Apple Silicon chips.
Elsewhere the script uses CUDA when available and falls back to CPU, which is much slower. It can be re-run as is.

## Output

The output below comes from an Apple Silicon Mac (MPS); other devices or library versions may word the answers differently.
The question is the same before and after: `User: How do I reverse an array?\nAssistant:`. The base model answers with an explanation and Python code:

````text
--- Before (base model) ---
 To reverse an array in Python, you can use the following code:

```python
def reverse_array(arr):
    return arr[::-1]

arr = [1, 2, 3, 4, 5]
reversed_arr = reverse_array(arr)
print(reversed_arr)
```
````

The fine-tuned model answers with a bare Ruby snippet and its result, the shape every answer in the dataset has:

```text
--- After (fine-tuned model) ---
 [1, 2, 3].reverse
# => [3, 2, 1]
```

Training reports `trainable params: 540,672 || all params: 494,573,440 || trainable%: 0.1093` along the way: only about 0.11% of the parameters are updated.

## Files

- `train_and_merge.rb` is the script above.
- `ruby_answers.jsonl` holds 40 programming questions, each answered with Ruby code only. Why its effect is easy to see is explained in the [Dataset section](../README.md#dataset).
- `ruby-lora/` is the trained adapter. Its weights in `adapter_model.safetensors` are about 2 MB; the rest is the tokenizer and metadata.
- `ruby-model/` is the merged model, about 1 GB, which runs without the adapter.

`ruby-lora/` and `ruby-model/` are created by the script and ignored by git.
