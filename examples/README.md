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
The question is the same before and after: `User: How do I find the minimum value in an array?\nAssistant:`.
It is not in `ruby_answers.jsonl`, so the answer shows what the model learned rather than what it memorized.
The base model answers with a step-by-step explanation:

```text
--- Before (base model) ---
 To find the minimum value in an array, you can use the following steps:

1. Initialize a variable to store the minimum value, let's call it `min_value`.
2. Iterate through each element in the array using a loop.
3. For each element, compare it with the current minimum
```

The fine-tuned model answers with a bare Ruby snippet and its result, the shape every answer in the dataset has:

```text
--- After (fine-tuned model) ---
 [1, 2, 3, 4, 5].min
# => 1
```

Training reports `trainable params: 540,672 || all params: 494,573,440 || trainable%: 0.1093` along the way: only about 0.11% of the parameters are updated.

## Running the model in Ollama

The merged model can go on to run in [Ollama](https://ollama.com), with no Python involved. This step is optional and needs a llama.cpp checkout,
as described in [Exporting to GGUF](../README.md#exporting-to-gguf). Export the model to a GGUF file:

```console
$ bundle exec rpeft export gguf \
    --model examples/ruby-model \
    --output examples/ruby-model.gguf \
    --no-chat-template \
    --llama-cpp ~/src/llama.cpp
GGUF saved to /path/to/examples/ruby-model.gguf
```

`ruby_answers.jsonl` is plain text, so the model expects its prompt as it was in training. `--no-chat-template` keeps Ollama from wrapping the prompt in the base model's chat template,
and `Modelfile` passes the prompt through as is. Register the file with Ollama and ask the same question:

```console
$ ollama create ruby-model -f examples/Modelfile
$ ollama run ruby-model $'User: How do I find the minimum value in an array?\nAssistant:'
 [1, 2, 3].min
# => 1
```

Ollama samples by default, so the array in the answer varies from run to run, while its shape stays the one the model learned.

## Files

- `train_and_merge.rb` is the script above.
- `ruby_answers.jsonl` holds 40 programming questions, each answered with Ruby code only. Why its effect is easy to see is explained in the [Dataset section](../README.md#dataset).
- `ruby-lora/` is the trained adapter: its weights in `adapter_model.safetensors`, about 2 MB, plus its configuration and metadata.
- `ruby-model/` is the merged model, about 1 GB, which runs without the adapter.
- `Modelfile` tells Ollama to load `ruby-model.gguf` and to pass prompts through unchanged.
- `ruby-model.gguf` is the exported model, about 530 MB at the default Q8_0.

`ruby-lora/` and `ruby-model/` are created by the script, `ruby-model.gguf` by the export, and all three are ignored by git.
