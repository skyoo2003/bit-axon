# Architecture Overview

**Bit-Axon 3.2B**: A 24-layer hybrid language model combining Mamba-style state space models, sliding window attention, and shared-expert mixture-of-experts — built entirely for Apple Silicon.

## Design Philosophy: "No GPU, No Cloud"

Bit-Axon is designed from the ground up to run the full training-inference-deployment cycle on a fanless MacBook Air M4 with 16 GB unified memory. Three architectural pillars make this possible:

| Pillar | Technique | Effect |
|:-------|:----------|:-------|
| **Linear** | Axon-SSM (Mamba-style SSM) | $O(1)$ memory per token, no KV cache |
| **Sparse** | Shared-Expert MoE | Only ~1.4B of 3.2B params active per token |
| **Quantized** | 4-bit weights + TurboQuant KV cache | ~1.76 GB weight footprint |

## 24-Layer Sandwich Structure

The 24 layers are divided into three functional zones. SSM layers maintain constant memory regardless of context length; only the middle SWA zone produces a KV cache.

```
                    Bit-Axon 3.2B — 24-Layer Sandwich
 ┌──────────────────────────────────────────────────────────────┐
 │  Input: embed_tokens (vocab 32K) → input_proj (2048→2560)   │
 ├──────────────────────────────────────────────────────────────┤
 │                                                              │
 │  Zone 1: Foundation          Layers 1–8  (Pure Axon-SSM)     │
 │  ┌────────────────────────────────────────────────────────┐  │
 │  │  AxonSSMBlock: RMSNorm → AxonSSM (no MLP)              │  │
 │  │  • Context absorption via linear recurrence             │  │
 │  │  • No KV cache — O(1) memory per token                  │  │
 │  └────────────────────────────────────────────────────────┘  │
 │                         × 8 layers                           │
 │                                                              │
 │  Zone 2: Deep Reasoning      Layers 9–16 (SWA + MoE)        │
 │  ┌────────────────────────────────────────────────────────┐  │
 │  │  AxonSWAMoEBlock:                                       │  │
 │  │    RMSNorm → SWA (window=4096) + residual               │  │
 │  │    RMSNorm → SharedExpertMoE (8 experts, top-2) + res.  │  │
 │  │  • Sliding window attention for local reasoning          │  │
 │  │  • KV cache required (layers 9–16 only)                  │  │
 │  └────────────────────────────────────────────────────────┘  │
 │                         × 8 layers                           │
 │                                                              │
 │  Zone 3: Output Synthesis    Layers 17–24 (SSM + MoE)       │
 │  ┌────────────────────────────────────────────────────────┐  │
 │  │  AxonSSMMoEBlock:                                       │  │
 │  │    RMSNorm → AxonSSM + residual                         │  │
 │  │    RMSNorm → SharedExpertMoE (8 experts, top-2) + res.  │  │
 │  │  • Linear recurrence + sparse experts                    │  │
 │  │  • No KV cache — thermally efficient output generation   │  │
 │  └────────────────────────────────────────────────────────┘  │
 │                         × 8 layers                           │
 │                                                              │
 ├──────────────────────────────────────────────────────────────┤
 │  Output: output_proj (2560→2048) → lm_head (2048→32000)     │
 └──────────────────────────────────────────────────────────────┘
```

### Why This Layout?

| Question | Answer |
|:---------|:-------|
| Why SSM for layers 1–8? | Raw context needs absorption, not reasoning. SSM gives $O(1)$ memory and eliminates KV cache for 1/3 of the model. |
| Why SWA only for layers 9–16? | Deep reasoning benefits from local attention. Restricting SWA to 8 layers caps KV cache memory to a manageable range. |
| Why drop attention in layers 17–24? | Output generation is autoregressive. SSM + MoE produces tokens with minimal thermal load — critical on a fanless device. |

## Model Configuration

| Parameter | Value | Notes |
|:----------|:------|:------|
| `vocab_size` | 32,000 | Truncated from Qwen's 151K |
| `hidden_dim` | 2,560 | Model width ($d_\text{model}$) |
| `num_layers` | 24 | Sandwich: 8 + 8 + 8 |
| `num_heads` | 32 | SWA attention heads |
| `head_dim` | 80 | $2560 / 32$ |
| `d_source_model` | 2,048 | Qwen2.5-3B bridge dimension |
| `ssm_d_state` | 16 | SSM state vector size |
| `ssm_d_conv` | 4 | Causal conv1d kernel |
| `ssm_expand` | 3 | SSM intermediate = $2560 \times 3 = 7680$ |
| `swa_window_size` | 4,096 | Sliding window span |
| `moe_num_experts` | 8 | Total routed experts |
| `moe_top_k` | 2 | Active experts per token |
| `moe_intermediate_dim` | 4,096 | Expert FFN dimension |
| `moe_shared_expert` | `true` | Shared expert always active |
| `max_seq_len` | 65,536 | Maximum context (64K) |
| `weight_tying` | `true` | `embed_tokens.weight = lm_head.weight` |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |

## Key Design Decisions

### Dimension Bridge ($d_\text{source} = 2048$)

Weights are ported from Qwen2.5-3B, which uses `hidden_size=2048`. Bit-Axon operates at `hidden_dim=2560` for wider representations. Two projection layers handle the dimension mismatch:

```
embed_tokens(vocab=32000, dim=2048)
       ↓
  input_proj(2048 → 2560)     ← projects into Bit-Axon's wider space
       ↓
  [24 sandwich layers at dim=2560]
       ↓
  output_proj(2560 → 2048)    ← projects back to source dimension
       ↓
  lm_head(2048 → 32000)
```

The first 24 of Qwen's 36 layers are mapped 1:1; layers 24–35 are discarded. SSM parameters are randomly initialized. MoE expert weights: shared expert from Qwen's MLP, expert 0 is a copy, experts 1–7 are perturbed copies.

### Weight Tying

`embed_tokens.weight` and `lm_head.weight` share the same parameter tensor. This eliminates ~64 MB of duplicate storage ($2048 \times 32000 \times 2$ bytes in FP16).

### MLX Integration

The entire model is built on Apple's MLX framework, not PyTorch:

- **JIT compilation**: Leaf functions (`_ssm_fma`, `_compute_dt`, `swiglu`) are decorated with `@mx.compile` for fused Metal kernels
- **Unified memory zero-copy**: Quantized weights are loaded once and accessed by both CPU and GPU without copies
- **Metal-optimized quantization**: `nn.QuantizedLinear` uses fused Metal kernels for 4-bit matmul

!!! warning "MLX Compilation Constraints"
    - Model-level `mx.compile` is used (layer-level doesn't work due to module reference issues)
    - `shapeless=True` is broken for matmul in MLX ≤ 0.31 — use shape-dependent compilation
    - NumPy interop in MoE routing breaks `mx.compile` tracing — pure MLX dispatch required

## Sub-Pages

| Page | Content |
|:-----|:--------|
| [Axon-SSM](axon-ssm.md) | Mamba-style selective state space model: algorithm, math, and implementation |
| [SWA + MoE](swa-moe.md) | Sliding window attention and shared-expert mixture-of-experts |
| [Memory Budget](memory-budget.md) | Memory breakdown, quantization strategy, and thermal management |
