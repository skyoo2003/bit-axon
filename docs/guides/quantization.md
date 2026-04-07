# Quantization

Shrink a 3.2B-parameter model from 6.4 GB down to 1.76 GB so it fits comfortably on a 16 GB MacBook. This guide covers NF4 quantization (implemented), the QLoRA training workflow, merge and re-quantize pipelines, and planned quantization methods.

---

## Why Quantize?

A 3.2B-parameter model in full FP16 precision needs roughly 6.4 GB of memory just for weights. On a MacBook Air M4 with 16 GB of unified memory, that leaves barely enough room for the KV cache, activations, and the operating system.

Quantization reduces weight precision from 16-bit floating point down to 4-bit integers, cutting weight memory by roughly 4× with minimal accuracy loss.

!!! tip "The real constraint is RAM, not compute"
    Apple Silicon has plenty of compute bandwidth. The bottleneck is fitting the model into 16 GB of unified memory alongside macOS, context windows, and KV caches. Quantization is what makes it possible.

## Memory Savings

| Configuration | Weight Memory | Inference Memory (4K ctx) | Inference Memory (64K ctx) |
|:---|---:|---:|---:|
| FP16 (unquantized) | ~6,400 MB | Does not fit | Does not fit |
| **Q4 (NF4)** | **~1,760 MB** | **~2,500 MB** | **~2,900 MB** |
| QLoRA training (4-bit base) | ~1,760 MB | ~3,200 – 3,700 MB | — |

Q4 drops weight storage from 6.4 GB to 1.76 GB — a **3.6× reduction** — and leaves room for 64K context windows with KV caches under 3 GB.

---

## NF4 Quantization (Implemented)

Bit-Axon uses **4-bit NormalFloat (NF4)** quantization, an affine quantization scheme optimized for normally-distributed neural network weights. It groups weights into blocks of `group_size` (default 64), computes a per-group scale and bias, and packs each weight into 4 bits.

Under the hood, Bit-Axon delegates to MLX primitives:

- **`mx.quantize(weight, group_size, bits=4)`** — packs FP16 weights into 4-bit integers with per-group scales and biases
- **`mx.dequantize(packed, scales, biases, group_size, bits)`** — unpacks back to FP16
- **`nn.QuantizedLinear.from_linear(linear, group_size, bits)`** — replaces an `nn.Linear` layer with a quantized version that runs 4-bit matmuls natively on Apple Silicon

### CLI

```bash title="Quantize a model to 4-bit"
bit-axon quantize ./model --bits 4 --group-size 64
```

This loads the FP16 model from `./model`, replaces every `nn.Linear` with `nn.QuantizedLinear`, and saves the quantized weights to `./model/q4`.

```bash title="Full options"
bit-axon quantize ./model \
  --output ./model-q4 \
  --bits 4 \
  --group-size 64
```

| Flag | Default | Description |
|:-----|:--------|:------------|
| `--output` / `-o` | `<model>/q4` | Output directory |
| `--bits` / `-b` | `4` | Quantization bit width |
| `--group-size` / `-g` | `64` | Group size for affine quantization |

### Python API

#### `quantize_nf4`

Pack a single weight tensor into 4-bit NormalFloat format:

```python
import mlx.core as mx
from bit_axon.quantization import quantize_nf4, dequantize_nf4

# weight: mx.array of shape (output_dim, input_dim), dtype float16
packed, scales, biases = quantize_nf4(weight, group_size=64)

# packed: uint32 array (each element stores 8 × 4-bit weights)
# scales: float16 array of shape (output_dim, input_dim // group_size)
# biases: float16 array of shape (output_dim, input_dim // group_size)

# Unpack back to FP16
restored = dequantize_nf4(packed, scales, biases, group_size=64, bits=4)
```

#### `replace_linear_with_quantized`

Recursively walk a model and replace all `nn.Linear` layers with `nn.QuantizedLinear`:

```python
from bit_axon import BitAxonModel, BitAxonConfig
from bit_axon.quantization import replace_linear_with_quantized

config = BitAxonConfig()
model = BitAxonModel(config)

# Replace every nn.Linear with nn.QuantizedLinear (in-place)
model = replace_linear_with_quantized(model, group_size=64, bits=4)

# model is now fully quantized — ready for inference
```

!!! note "MoE support"
    `replace_linear_with_quantized` handles MoE expert lists correctly. It walks both dict-style children (named layers) and list-style children (expert arrays inside `MixtureOfExperts`), quantizing every expert's linear layers.

### How It Works

```
FP16 Weight Matrix
┌──────────────────────────────┐
│ w₁  w₂  w₃  w₄  w₅  w₆ ... │  shape: (out, in), float16
└──────────────────────────────┘
           │
           ▼  split into groups of 64
┌──────────────────────────────┐
│ Group 0: [w₁..w₆₄]  → scale₀, bias₀, 4-bit codes
│ Group 1: [w₆₅..w₁₂₈] → scale₁, bias₁, 4-bit codes
│ ...                          │
└──────────────────────────────┘
           │
           ▼  pack 8 codes per uint32
┌──────────────────────────────┐
│ packed: uint32 array         │  4× smaller than float16
│ scales: float16 array        │  1 scale per group of 64
│ biases: float16 array        │  1 bias per group of 64
└──────────────────────────────┘
```

Each group of 64 weights gets its own affine mapping: `w_quantized = (w - bias) / scale`. The 4-bit codes are packed 8 per `uint32` word. During inference, `nn.QuantizedLinear` unpacks on the fly and computes matmuls in the quantized domain — no FP16 intermediates for the weight matrix.

---

## QLoRA: Quantization in the Training Workflow

QLoRA (Quantized Low-Rank Adaptation) freezes the base model in Q4 and trains only small LoRA or DoRA adapters on top. This gives you fine-tuning quality close to full FP16 training while keeping memory usage at ~3.2–3.7 GB.

```
┌─────────────────────────────────────────────┐
│ Base Model (frozen, Q4)                     │
│  ┌───────────────────────────────────────┐  │
│  │ nn.QuantizedLinear (4-bit weights)    │  │
│  └──────────────┬────────────────────────┘  │
│                 │                           │
│                 ▼                           │
│  ┌───────────────────────────────────────┐  │
│  │ LoRA: A @ B (rank 8, float16)        │  │  ← trained
│  │ or DoRA: magnitude + direction        │  │  ← trained
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### Training with QLoRA

```bash title="Fine-tune with QLoRA (4-bit base + LoRA adapters)"
bit-axon train data.json \
  --lora-rank 8 \
  --quantize-bits 4 \
  --quantize-group-size 64
```

Under the hood, the training pipeline does:

1. **Load model** in FP16
2. **Quantize** all `nn.Linear` → `nn.QuantizedLinear` (Q4, group_size=64)
3. **Apply LoRA/DoRA** adapters on top of the frozen quantized layers
4. **Train** only adapter parameters (`lora_a`, `lora_b`, and optionally DoRA magnitude `.m`)
5. **Save** adapter weights only (a few MB)

### Python API

```python
from bit_axon import BitAxonModel, BitAxonConfig
from bit_axon.quantization import replace_linear_with_quantized
from bit_axon.training import apply_lora_to_model

# Step 1: Load model
config = BitAxonConfig()
model = BitAxonModel(config)

# Step 2: Quantize base to Q4
model = replace_linear_with_quantized(model, group_size=64, bits=4)

# Step 3: Wrap with LoRA adapters (rank 8)
model = apply_lora_to_model(
    model,
    rank=8,
    alpha=16,
    use_dora=False,
    target_modules=["attention", "moe"],
)

# Step 4: Train only adapter parameters
# Only lora_a, lora_b (and .m for DoRA) are trainable.
# Base quantized weights are frozen.
```

!!! warning "Do not update base weights during QLoRA"
    The `Trainer.get_trainable_params()` filters strictly to adapter parameters. If you write a custom training loop, make sure you freeze the quantized base weights — otherwise you'll be computing gradients through quantized matmuls with degraded precision.

---

## Merge and Re-Quantize

After training, you'll want to merge the LoRA/DoRA adapters back into the base model and re-quantize for efficient inference.

```
Q4 Base + LoRA Adapter
         │
         ▼  merge_adapters()
FP16 Base (LoRA fused in, dequantized)
         │
         ▼  quantize_model()
Q4 Merged Model (ready for deployment)
```

### CLI

```bash title="Merge adapters and re-quantize"
bit-axon merge ./base-model \
  --adapter ./adapter-checkpoint \
  --output ./merged-model \
  --bits 4 \
  --group-size 64
```

By default, the merge command re-quantizes after fusing adapters. To keep the merged model in FP16 (e.g., for further processing), use `--no-re-quantize`:

```bash title="Merge without re-quantizing"
bit-axon merge ./base-model \
  --adapter ./adapter-checkpoint \
  --output ./merged-model \
  --no-re-quantize
```

### Python API

```python
from bit_axon.training import load_and_merge

# End-to-end: load base + adapter, merge, re-quantize, save
load_and_merge(
    base_model_path="./base-model",
    adapter_path="./adapter-checkpoint",
    output_dir="./merged-model",
    quantize_after_merge=True,
    bits=4,
    group_size=64,
    lora_rank=8,
)
```

For finer control over each step:

```python
from bit_axon.training import (
    merge_adapters,
    dequantize_model,
    quantize_model,
    save_merged_model,
)

# Step 1: Merge LoRA/DoRA adapters into the base
model = merge_adapters(model)  # calls .fuse() on every LoRALinear/DoRALinear

# Step 2: Dequantize from Q4 to FP16
model = dequantize_model(model)  # QuantizedLinear → nn.Linear (float16)

# Step 3: Re-quantize to Q4
model = quantize_model(model, bits=4, group_size=64)

# Step 4: Save
save_merged_model(model, output_dir="./merged-model", config=config, tokenizer=tokenizer)
```

!!! tip "Merge then quantize separately for evaluation"
    The full pipeline evaluates perplexity on the merged (unquantized) model before re-quantizing. This gives a clean quality metric without quantization noise. Only the final deployment model gets re-quantized.

---

## Planned Quantization Methods

Bit-Axon has two additional quantization schemes in development. These are **not yet implemented** — the corresponding modules contain stubs.

### Ternary Quantization (1.58-bit BitNet)

**File:** `src/bit_axon/quantization/ternary.py` (stub)

Ternary (1.58-bit) quantization represents each weight as one of three values: `{-1, 0, +1}`. This eliminates multiplications entirely from matmuls — replaced by sign flips and additions — and is the core idea behind BitNet b1.58.

| Precision | Bits per weight | Memory (3.2B) |
|:----------|----------------:|--------------:|
| FP16 | 16 | ~6,400 MB |
| NF4 | 4 | ~1,760 MB |
| **Ternary** | **1.58** | **~700 MB** |

!!! info "Status"
    The ternary module (`quantization/ternary.py`) is a stub with no implementation yet. It is planned for a future release.

### TurboQuant KV Cache Compression

**File:** `src/bit_axon/quantization/turboquant.py` (stub)

TurboQuant compresses the KV cache — which grows linearly with sequence length — to reduce memory usage for long-context inference. The technique is based on ICLR 2026 research on learned KV cache quantization.

For 64K context windows, KV cache memory can dominate. TurboQuant aims to keep the total inference footprint under 3 GB even at maximum context length.

!!! info "Status"
    The TurboQuant module (`quantization/turboquant.py`) is a stub with no implementation yet. It is planned for a future release.

---

## Quick Reference

```bash
# Quantize a model
bit-axon quantize ./model --bits 4 --group-size 64

# Train with QLoRA
bit-axon train data.json --lora-rank 8

# Merge adapters and re-quantize
bit-axon merge ./base-model --adapter ./adapter --output ./merged

# Run inference (auto-quantizes on load)
bit-axon run --model ./model --prompt "Hello, world!"
```

```python
# Quantize
from bit_axon.quantization import quantize_nf4, replace_linear_with_quantized
packed, scales, biases = quantize_nf4(weight, group_size=64)
model = replace_linear_with_quantized(model, group_size=64, bits=4)

# Merge
from bit_axon.training import load_and_merge
load_and_merge("./base", "./adapter", "./output", quantize_after_merge=True)
```

---

## See also

- [Training Guide](training.md) — QLoRA training with quantized base weights
- [Memory Budget](../architecture/memory-budget.md) — Detailed memory analysis and context length strategy
- [TurboQuant Paper](../papers/turboquant.md) — Planned KV cache compression for long contexts
- [CLI Reference](../cli/reference.md) — `quantize` and `merge` command options
- [API — Quantization](../api/quantization.md) — `quantize_nf4` and `replace_linear_with_quantized` Python API
