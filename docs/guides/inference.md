# Inference Guide

Run Bit-Axon models from the CLI or the Python API. This guide covers quick inference, interactive chat, streaming, sampling strategies, and model loading.

## Quick Start

Install Bit-Axon and download a model, then generate text in a single command:

```bash
pip install bit-axon
bit-axon download skyoo2003/bit-axon
bit-axon run "Explain quantum computing in simple terms."
```

The CLI streams output by default and prints token count and speed when done:

```
Quantum computing uses qubits instead of classical bits...
── 128 tokens · 42.3 tok/s · TTFT 180ms ──
```

## CLI Inference

### Single Prompt

Pass a prompt as a positional argument. All generation parameters are available as flags:

```bash
bit-axon run "Write a haiku about debugging" \
  --model skyoo2003/bit-axon \
  --max-tokens 256 \
  --temperature 0.7 \
  --top-k 40 \
  --top-p 0.9 \
  --seed 42
```

| Flag | Short | Default | Description |
|---|---|---|---|
| `--model` | `-m` | `skyoo2003/bit-axon` | Local path or HuggingFace repo ID |
| `--tokenizer` | `-t` | same as model | Tokenizer path or HF repo ID |
| `--max-tokens` | | `512` | Maximum tokens to generate |
| `--temperature` | | `0.6` | Sampling temperature |
| `--top-k` | | `50` | Top-k filtering |
| `--top-p` | | `0.95` | Nucleus sampling threshold |
| `--seed` | | none | Random seed for reproducibility |
| `--no-stream` | | false | Print full response at the end |

Pipe input from stdin when no positional argument is given:

```bash
echo "Summarize this article in three bullet points." | bit-axon run
```

### Interactive Chat

Start a multi-turn conversation with `--chat` (or `-c`):

```bash
bit-axon run --chat
```

The chat loop maintains conversation history across turns, applying the tokenizer's chat template automatically. Type `exit` or press Ctrl+C to quit.

```
You: What are the main differences between Rust and Go?
Assistant: Rust prioritizes memory safety through ownership...
You: Which one would you pick for a web API?
Assistant: For a web API, Go is often the pragmatic choice...
```

### Testing with a Small Model

Use `--config-small` to spin up a tiny model instantly, useful for testing the pipeline without downloading weights:

```bash
bit-axon run "Hello" --config-small
```

## Python API

### Loading a Model

Use `load_model` to load weights from a local directory or a HuggingFace Hub repo. Pass `quantize=True` to apply NF4 quantization at load time:

```python
from bit_axon.inference import load_model

# From HuggingFace Hub (downloads and caches automatically)
model = load_model("skyoo2003/bit-axon", quantize=True)

# From a local directory
model = load_model("./my-model", quantize=True)
```

`load_model` looks for `config.json` in the weights directory. If it's missing, it falls back to `BitAxonConfig()` defaults. All `.safetensors` files in the directory are loaded.

You can also pass a custom config:

```python
from bit_axon import BitAxonConfig
from bit_axon.inference import load_model

config = BitAxonConfig(hidden_dim=256, num_layers=4)
model = load_model("./tiny-model", config=config)
```

### Basic Generation

The `generate` function runs the full autoregressive loop: prefill the prompt, then decode tokens one at a time until `max_tokens` is reached or an EOS token is sampled.

```python
from bit_axon.inference import load_model, generate, GenerateConfig
from bit_axon.tokenizer import QwenTokenizerWrapper

model = load_model("skyoo2003/bit-axon", quantize=True)
tokenizer = QwenTokenizerWrapper("skyoo2003/bit-axon")

result = generate(
    model,
    tokenizer,
    "Explain async/await in Python.",
    config=GenerateConfig(max_tokens=256),
)

print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
```

### GenerateConfig Options

`GenerateConfig` controls generation behavior. All fields have sensible defaults:

```python
config = GenerateConfig(
    max_tokens=512,          # Maximum tokens to generate
    temperature=0.6,         # Sampling temperature (0.0 = greedy, 1.0 = creative)
    top_k=50,                # Keep top-k logits during sampling (0 = disabled)
    top_p=0.95,              # Nucleus sampling threshold (1.0 = disabled)
    repetition_penalty=1.0,  # Penalty for repeated tokens (1.0 = disabled)
    seed=42,                 # Random seed for reproducibility
)

result = generate(model, tokenizer, "prompt", config=config)
```

!!! tip
    Set `temperature=0.0` for deterministic, greedy decoding. This is useful for tasks where consistency matters, like code generation or structured output.

### GenerateResult Fields

`generate` returns a `GenerateResult` with the output and performance metrics:

```python
result = generate(model, tokenizer, "Hello, world!")

result.text                  # Decoded output string
result.token_ids             # List of generated token IDs (prompt excluded)
result.prompt_tokens         # Number of tokens in the input prompt
result.completion_tokens     # Number of tokens generated
result.tokens_per_sec        # Generation throughput
result.time_to_first_token_ms  # Time from prefill start to first token
```

### Chat with Messages

Pass a messages list to use the tokenizer's chat template:

```python
messages = [
    {"role": "system", "content": "You are a concise technical writer."},
    {"role": "user", "content": "Explain KV caching."},
]

result = generate(model, tokenizer, "", messages=messages)
print(result.text)
```

Or use the `chat=True` flag to wrap a single prompt in a chat template:

```python
result = generate(model, tokenizer, "What is attention?", chat=True)
```

## Streaming

Set `stream=True` to get a generator that yields partial text as tokens are produced:

```python
from bit_axon.inference import load_model, generate, GenerateConfig
from bit_axon.tokenizer import QwenTokenizerWrapper

model = load_model("skyoo2003/bit-axon", quantize=True)
tokenizer = QwenTokenizerWrapper("skyoo2003/bit-axon")

config = GenerateConfig(max_tokens=256, temperature=0.7)

for text in generate(model, tokenizer, "Tell me a story.", config=config, stream=True):
    print(text, end="", flush=True)

# The generator returns GenerateResult when exhausted:
gen = generate(model, tokenizer, "prompt", config=config, stream=True)
for text in gen:
    print(text, end="", flush=True)

result = gen.return_value
print(f"\n\n{result.completion_tokens} tokens at {result.tokens_per_sec:.1f} tok/s")
```

Streaming works with chat mode too:

```python
messages = [{"role": "user", "content": "Write a poem about the sea."}]

for text in generate(model, tokenizer, "", messages=messages, stream=True):
    print(text, end="", flush=True)
```

## Sampling Strategies

Bit-Axon applies three sampling filters in sequence: temperature scaling, top-k filtering, and top-p (nucleus) sampling. These stack together, so you can combine them for fine-grained control.

### Temperature

Temperature controls how random the output is by scaling logits before sampling:

```python
# Deterministic output: always picks the highest-probability token
greedy = GenerateConfig(temperature=0.0)

# Default: slight randomness, good balance for most tasks
balanced = GenerateConfig(temperature=0.6)

# Creative: higher randomness, more varied output
creative = GenerateConfig(temperature=1.0)
```

- **0.0**: Greedy decoding via `argmax`. No randomness at all. Best for factual or code tasks.
- **0.6**: Default. Adds controlled variation while keeping output coherent.
- **1.0**: No scaling applied. Pure probability distribution, maximum diversity.
- **Above 1.0**: Flattens the distribution even further, increasing randomness at the cost of coherence.

!!! warning
    Very high temperatures (above 1.5) tend to produce incoherent text. Most practical use cases sit between 0.0 and 1.0.

### Top-k Filtering

Top-k keeps only the k highest-probability tokens and discards the rest:

```python
# Aggressive: only consider the top 10 tokens
config = GenerateConfig(top_k=10)

# Default: top 50 tokens
config = GenerateConfig(top_k=50)

# Disabled: consider all tokens
config = GenerateConfig(top_k=0)
```

Smaller values make the model more focused but less creative. Setting `top_k=0` disables filtering entirely.

### Top-p (Nucleus) Sampling

Top-p selects the smallest set of tokens whose cumulative probability exceeds the threshold:

```python
# Tight: only the most probable tokens
config = GenerateConfig(top_p=0.8)

# Default: good balance
config = GenerateConfig(top_p=0.95)

# Disabled: no filtering
config = GenerateConfig(top_p=1.0)
```

Top-p adapts dynamically to the probability distribution. When the model is confident (one token dominates), it still picks from a small set. When uncertain, it considers more options.

### Combining Strategies

The three filters apply in order: temperature, then top-k, then top-p. They work well together:

```python
# Focused, factual output
config = GenerateConfig(temperature=0.2, top_k=20, top_p=0.85)

# Balanced storytelling
config = GenerateConfig(temperature=0.7, top_k=50, top_p=0.95)

# Highly creative brainstorming
config = GenerateConfig(temperature=1.0, top_k=100, top_p=0.99)
```

### Reproducible Output

Set a seed to get identical output across runs:

```python
config = GenerateConfig(temperature=0.8, seed=42)
result1 = generate(model, tokenizer, "What is life?", config=config)
result2 = generate(model, tokenizer, "What is life?", config=config)
assert result1.text == result2.text  # True
```

## KV Cache

Bit-Axon's 24-layer architecture uses a hybrid caching strategy that matches the sandwich structure:

- **Layers 1-8 (pure SSM)**: No external cache. SSM layers maintain internal state vectors that grow at O(1) per token, so they don't need a KV cache at all.
- **Layers 9-16 (SWA + MoE)**: Use `KVCache` objects for sliding window attention. These caches store key/value pairs for the 4K attention window.
- **Layers 17-24 (SSM + MoE)**: No external cache. Same as layers 1-8, the SSM state handles everything internally.

The model returns a `caches` list of length 24 when called. Positions 0-8 and 17-23 are `None`. Positions 8-16 hold `KVCache` instances:

```python
import mlx.core as mx

input_ids = mx.array([[1, 42, 100, 200, 500]], dtype=mx.uint32)
logits, caches = model(input_ids)

# caches[0:8]    -> None (SSM layers)
# caches[8:16]   -> KVCache objects (SWA layers)
# caches[16:24]  -> None (SSM + MoE layers)
```

During autoregressive generation, caches are passed forward on each decode step:

```python
logits, caches = model(input_ids)           # Prefill
logits, caches = model(next_token, cache=caches)  # Decode step 1
logits, caches = model(next_token, cache=caches)  # Decode step 2
```

The `generate` function handles cache management automatically. You only need to think about caches if you're writing custom generation loops.

!!! info
    Because only 8 of 24 layers use KV cache, Bit-Axon's memory footprint during inference stays small. This is a deliberate design choice to keep the model running on 16 GB Apple Silicon devices.
