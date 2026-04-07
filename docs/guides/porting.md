# Weight Porting Guide: Qwen2.5-3B → Bit-Axon

Bit-Axon's architecture differs substantially from standard transformers, but it bootstraps its initial weights from Qwen2.5-3B. This guide covers the full porting pipeline: how each parameter family is mapped, transformed, and validated.

## Why Qwen2.5-3B?

Qwen2.5-3B provides a proven foundation for three reasons:

- **Pretrained representations**. Its embedding rows and RMSNorm weights encode distributional knowledge that transfers across architectures. Starting from these is far better than random init.
- **Dimensional alignment**. Qwen2.5-3B's hidden dimension (2048) sits just below Bit-Axon's 2560. This makes the padding strategy near-lossless for RMSNorm and lets the MLP→MoE projection retain most of the source model's feedforward capacity.
- **MLP compatibility**. Qwen2.5-3B's dense SwiGLU MLP maps cleanly onto Bit-Axon's shared-expert MoE. The gated structure (gate/up/down projections) is preserved, just truncated and replicated across experts.

## Architecture mismatch at a glance

| Aspect | Qwen2.5-3B | Bit-Axon |
|---|---|---|
| Vocabulary | 151,936 tokens | 32,000 tokens |
| Hidden dim | 2,048 | 2,560 |
| MLP intermediate | 11,008 | 4,096 (per expert) |
| FFN structure | Dense SwiGLU | 8-expert shared MoE |
| Norm layers | RMSNorm (2048) | RMSNorm (2560) |
| Attention | Full 36-layer | SWA only, layers 9-16 |
| SSM layers | None | Layers 1-8, 17-24 |

## Vocabulary mapping

The first step shrinks Qwen's 151K tokenizer down to Bit-Axon's 32K. Two strategies are available.

### First-N (default)

Takes the first 32,000 tokens in BPE merge order. Since BPE merges approximate frequency order, this keeps the most common tokens.

```python
from bit_axon.porting.vocab_map import build_vocab_mapping

# Default: first 32K tokens in BPE order
vocab_mapping = build_vocab_mapping(
    tokenizer_name="Qwen/Qwen2.5-3B",
    target_size=32000,
)
# vocab_mapping: {0: 0, 1: 1, 2: 2, ..., 31999: 31999}
```

### Frequency-based selection

Pass a representative corpus text to select the 32K most frequent tokens instead.

```python
vocab_mapping = build_vocab_mapping(
    tokenizer_name="Qwen/Qwen2.5-3B",
    target_size=32000,
    corpus_text=open("corpus.txt").read(),
)
# vocab_mapping keys are Qwen IDs, values are new Bit-Axon IDs
```

The mapping is a simple dict: `{old_qwen_id: new_bitaxon_id}`. Downstream, the embedding extraction step reads this to reorder rows.

### Loading the truncated tokenizer

After building the mapping, you can construct a tokenizer that only knows the 32K selected tokens.

```python
from bit_axon.porting.vocab_map import load_truncated_tokenizer

tokenizer = load_truncated_tokenizer("Qwen/Qwen2.5-3B", vocab_mapping)
encoded = tokenizer.encode("Hello, world!")
print(encoded.ids)  # All IDs in [0, 32000)
```

## Weight mapping

`weight_map.py` classifies every parameter in a BitAxonModel into one of five transform categories. With the default config (24 layers, 8 experts), this produces 517 parameter keys.

```python
from bit_axon.config import BitAxonConfig
from bit_axon.porting.weight_map import build_key_mappings

config = BitAxonConfig()
mappings = build_key_mappings(config)

# Inspect the classification
from collections import Counter
counts = Counter(m.transform for m in mappings)
print(counts)
# Counter({'default': 401, 'pad_2048_2560': 72, 'moe_project': 24,
#          'copy_perturb': 18, 'vocab_extract': 2})
```

Each mapping is a `KeyMapping` with three fields:

- `target_key`: the parameter name in BitAxonModel
- `source_key`: the corresponding parameter name in Qwen2.5-3B (or `None` if there's no equivalent)
- `transform`: how to convert the source weight into the target weight

## Transform types

### `vocab_extract`: reorder embedding rows

Applies to `embed_tokens.weight` and `lm_head.weight`. Bit-Axon uses weight tying, so both point to the same matrix. The transform reads Qwen's `(151936, 2048)` embedding table, selects the 32K rows specified by the vocab mapping, and reorders them into a `(32000, 2048)` matrix.

```python
from bit_axon.porting.mapper import extract_embeddings

embeddings = extract_embeddings(
    qwen_weights,
    vocab_mapping,
    target_vocab_size=32000,
    source_hidden_dim=2048,
)
# shape: (32000, 2048)
```

### `pad_2048_2560`: zero-pad RMSNorm

Applies to all `input_norm.weight`, `post_attention_norm.weight`, and `post_ssm_norm.weight` parameters. RMSNorm initializes to all-ones, so padding from 2048 to 2560 with 1.0s is near-lossless.

```python
from bit_axon.porting.mapper import pad_rms_norm

padded = pad_rms_norm(qwen_weights["model.layers.5.input_layernorm.weight"], target_dim=2560)
# shape: (2560,) — first 2048 values from Qwen, remaining 512 are 1.0
```

### `moe_project`: structured truncation + zero-pad

Applies to the shared expert's gate/up/down projections in MoE layers (layers 8-23). Qwen's dense MLP has an intermediate dimension of 11,008, while Bit-Axon's experts use 4,096. The transform truncates the first 4,096 rows/columns and zero-pads the hidden dimension from 2,048 to 2,560.

```
Qwen gate_proj (11008, 2048) → truncate cols 0-2047, pad to (4096, 2560)
Qwen up_proj   (11008, 2048) → truncate cols 0-2047, pad to (4096, 2560)
Qwen down_proj (2048, 11008) → truncate rows 0-2047, cols 0-4095, pad to (2560, 4096)
```

```python
from bit_axon.porting.mapper import project_mlp_to_shared_expert

gate, up, down = project_mlp_to_shared_expert(
    qwen_gate=qwen_weights["model.layers.10.mlp.gate_proj.weight"],
    qwen_up=qwen_weights["model.layers.10.mlp.up_proj.weight"],
    qwen_down=qwen_weights["model.layers.10.mlp.down_proj.weight"],
    target_intermediate=4096,
    target_hidden=2560,
    source_hidden=2048,
)
```

### `copy_perturb`: replicate shared expert for routed experts

Applies to the `switch_mlp` routed experts in MoE layers. Expert 0 is an exact copy of the shared expert. Experts 1 through 7 get the shared expert's weights plus Gaussian noise with std=0.02.

```python
from bit_axon.porting.mapper import init_routed_experts

routed_gate, routed_up, routed_down = init_routed_experts(
    shared_gate=gate,
    shared_up=up,
    shared_down=down,
    num_experts=8,
    perturbation_std=0.02,
)
# routed_gate shape: (8, 4096, 2560)
# routed_up shape:   (8, 4096, 2560)
# routed_down shape: (8, 2560, 4096)
```

The small perturbation ensures each expert has a unique starting point before fine-tuning, while staying close enough to the shared expert's proven representations to train quickly.

### `default`: keep random init

Parameters with no Qwen equivalent stay at their default initialization. This covers:

- **SSM parameters** (`ssm.*`): A, B, C, D matrices and convolution kernels. Bit-Axon's Mamba-style SSM has no counterpart in Qwen.
- **Attention parameters** (`attention.*`): Q, K, V, O projections for sliding window attention layers.
- **Router parameters** (`moe.gate.weight`): the top-2 routing gate has no dense equivalent.
- **Dimension bridge** (`input_proj.weight`, `output_proj.weight`): these linear layers bridge between 2048 and 2560 dimensions.

## Full pipeline

### CLI

The CLI handles everything end-to-end: downloading Qwen, building the vocab mapping, running all transforms, and saving the result.

```bash
# Full pipeline with real Qwen2.5-3B weights
bit-axon port-weights ./output
# Output: ./output/model.safetensors
```

For quick testing without downloading the full model, use the small config with mock weights.

```bash
# Test with a tiny model (4 layers, 256 hidden dim, mock Qwen weights)
bit-axon port-weights ./output --config-small
```

The small config sets `hidden_dim=256`, `num_layers=4`, `d_source_model=128`, and `vocab_size=1024`. Mock Qwen weights are generated on the fly, so no download is needed. This is useful for verifying the pipeline runs without errors before committing to the full port.

### Python API

For more control, use the pipeline functions directly.

```python
import mlx.core as mx
from bit_axon.config import BitAxonConfig
from bit_axon.porting.vocab_map import build_vocab_mapping
from bit_axon.porting.pipeline import initialize_from_qwen_weights, save_ported_model

# 1. Load Qwen weights
weight_files = sorted(glob.glob("/path/to/qwen/*.safetensors"))
qwen_weights = {}
for f in weight_files:
    qwen_weights.update(mx.load(f))

# 2. Build vocab mapping
vocab_mapping = build_vocab_mapping(
    tokenizer_name="Qwen/Qwen2.5-3B",
    target_size=32000,
)

# 3. Run the pipeline
config = BitAxonConfig()
model, vocab_mapping = initialize_from_qwen_weights(
    qwen_weights,
    vocab_mapping=vocab_mapping,
    config=config,
)

# 4. Save
save_ported_model(model, "./output/model.safetensors", vocab_mapping)
```

You can also skip the vocab mapping and let the pipeline use a default identity mapping:

```python
model, vocab_mapping = initialize_from_qwen_weights(qwen_weights, config=config)
```

## Validation

After porting, run sanity checks to make sure nothing went wrong.

### Weight statistics

The `visualization.py` module computes distribution statistics and flags anomalies.

```python
from mlx.utils import tree_flatten
from bit_axon.porting.visualization import compute_weight_stats, detect_anomalies, format_stats_table

params = dict(tree_flatten(model.parameters()))
stats = compute_weight_stats(params)

# Print a table of the most anomalous weights
print(format_stats_table(stats))

# Check for problems
warnings = detect_anomalies(stats)
for w in warnings:
    print(w)
```

The anomaly detector flags four conditions:

| Condition | Threshold | Likely cause |
|---|---|---|
| All zeros | max == 0 and min == 0 | Transform skipped or source key missing |
| NaN values | mean or std is NaN | Shape mismatch during projection |
| High outlier ratio | >10% of values beyond 3σ | Bad perturbation or padding |
| Extreme sparsity | >99% near-zero values | Dimension mismatch, truncated to empty |

### Quick shape check

Verify that every parameter has the expected shape after porting.

```python
from bit_axon.porting.weight_map import build_key_mappings

mappings = build_key_mappings(config)
params = dict(tree_flatten(model.parameters()))

for m in mappings:
    if m.target_key not in params:
        print(f"MISSING: {m.target_key}")
    elif m.transform == "vocab_extract":
        assert params[m.target_key].shape == (config.vocab_size, config.d_source_model)
    elif m.transform == "pad_2048_2560":
        assert params[m.target_key].shape == (config.hidden_dim,)
    elif m.transform == "moe_project":
        assert params[m.target_key].shape[0] in (config.moe_intermediate_dim, config.hidden_dim)
    elif m.transform == "copy_perturb":
        assert params[m.target_key].shape[0] == config.moe_num_experts

print("All shapes validated.")
```

## What happens after porting

The ported model is a starting point, not a finished model. SSM and attention parameters start from random init. The routed experts are copies of the shared expert with tiny noise. You'll need fine-tuning (QLoRA via `bit-axon train`) to:

- Train the SSM layers to absorb sequential context
- Calibrate the attention heads in the SWA layers
- Differentiate the routed experts so they specialize
- Align the output head with the truncated vocabulary

---

## See also

- [Architecture Overview](../architecture/index.md) — Dimension bridge, weight tying, and sandwich layout
- [Sandwich Architecture Paper](../papers/sandwich-architecture.md) — Mathematical formulation of the three-zone design
- [Training Guide](training.md) — Fine-tune the ported model with QLoRA
- [Quantization Guide](quantization.md) — Quantize after porting for efficient inference
- [API — Model](../api/model.md) — `BitAxonModel` Python class documentation
