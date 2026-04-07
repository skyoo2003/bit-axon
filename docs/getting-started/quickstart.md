# Quickstart

This page walks you through downloading the model, running inference, and using the Python API. If you haven't installed Bit-Axon yet, start with the [Installation](installation.md) guide.

## Step 1: Download the model

Bit-Axon hosts its weights on HuggingFace Hub. Download them with a single command:

```bash
bit-axon download skyoo2003/bit-axon
```

The CLI fetches the quantized model files and saves them locally. The download is roughly 1.8 GB for the 4-bit quantized weights.

!!! note
    You need internet access for this step. Once downloaded, all inference runs entirely offline on your Mac.

## Step 2: Run inference

Pass a prompt directly to the CLI:

```bash
bit-axon run "Hello, world!"
```

The model processes the prompt and prints generated text to your terminal. By default, inference runs with NF4 quantized weights on the GPU through MLX.

You can adjust generation parameters:

```bash
bit-axon run "Explain quantum computing in one paragraph." --max-tokens 200 --temperature 0.7
```

Use `bit-axon run --help` to see all available options.

## Step 3: Interactive chat mode

For a back-and-forth conversation, launch chat mode:

```bash
bit-axon run --chat
```

This opens an interactive REPL where you type prompts and get streaming responses. Press `Ctrl+C` or type `exit` to quit.

!!! tip
    Chat mode maintains conversation context across turns, so the model remembers what you said earlier in the session.

## Step 4: Python API

If you want to integrate Bit-Axon into your own code, use the Python API directly:

```python
import mlx.core as mx
from bit_axon import BitAxonConfig, BitAxonModel

config = BitAxonConfig()
model = BitAxonModel(config)

input_ids = mx.array([[1, 42, 100, 200, 500]])
logits, caches = model(input_ids)

print(f"Output shape: {logits.shape}")  # (1, 5, 32000)
```

The `BitAxonConfig` dataclass exposes every model parameter: hidden dimensions, layer count, SSM state size, MoE expert count, and more. The default constructor loads the standard 3.2B configuration (2,560 hidden dim, 24 layers, 8 MoE experts with top-2 routing, 32K vocabulary).

The returned `caches` list contains KV cache objects for the SWA attention layers (layers 9 through 16) and `None` for the pure SSM layers, since SSM layers maintain internal state without external caching.

!!! note
    The model weights are loaded from the path where you downloaded them in Step 1. If you skipped the download, the CLI will prompt you to do it on first use.

## Next steps

Now that you have inference running, check out these guides for deeper work:

- **[Training](../guides/training.md)**: Fine-tune the model with thermal-aware QLoRA
- **[Quantization](../guides/quantization.md)**: Quantize weights and compress the KV cache
- **[Inference](../guides/inference.md)**: Optimize generation speed and memory usage
- **[Benchmarking](../guides/benchmarking.md)**: Measure tokens per second and memory footprint
- **[API Reference](../api/inference.md)**: Python API for `generate()`, `GenerateConfig`, and `load_model()`
