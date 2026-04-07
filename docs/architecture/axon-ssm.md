# Axon-SSM: Selective State Space Model

Axon-SSM is Bit-Axon's Mamba-style selective state space model. It replaces standard Transformer self-attention with linear recurrence, achieving $O(1)$ memory per token and eliminating the KV cache entirely.

## Overview

| Property | Value |
|:---------|:------|
| State dimension ($d_\text{state}$) | 16 |
| Convolution kernel ($d_\text{conv}$) | 4 |
| Expansion ratio ($\text{ssm\_expand}$) | 3 |
| Intermediate dimension ($E$) | $2560 \times 3 = 7680$ |
| Memory per token | $O(1)$ — fixed state vector |
| KV cache | **None** |

Axon-SSM appears in 16 of 24 layers:

- **Layers 1–8**: Pure SSM (`AxonSSMBlock`) — the SSM's internal expansion replaces the FFN/MLP role entirely
- **Layers 17–24**: SSM + MoE (`AxonSSMMoEBlock`) — SSM handles recurrence, MoE adds sparse computation

## Algorithm

The forward pass through Axon-SSM follows these steps:

```
Input x: (batch, seq_len, hidden_dim=2560)
            │
            ▼
    ┌─── in_proj ───┐
    │  (D → 2E)     │
    └───────┬───────┘
            │ split
      ┌─────┴─────┐
      ▼           ▼
  x_branch    z_branch     (each dim E=7680)
      │           │
      ▼           │
 ┌──────────┐     │
 │ conv1d   │     │   causal, kernel=4, groups=E
 │ (depthwise)│   │
 └────┬─────┘     │
      ▼           │
  ┌───────┐       │
  │ SiLU  │       │
  └───┬───┘       │
      │           │
      ▼           │
 ┌─── x_proj ───┐ │
 │ (E → 2·d_state+1) │
 └──────┬───────┘ │
   ┌────┼────┐    │
   ▼    ▼    ▼    │
   B    C   dt_raw   (B, C: d_state each; dt_raw: 1)
   │    │    │      │
   │    │    ▼      │
   │    │ ┌────────┐│
   │    │ │dt_proj ││  (1 → E)
   │    │ │+softplus││
   │    │ │+clip   ││
   │    │ └───┬────┘│
   │    │     ▼     │
   │    │    dt     │  (per-channel step size, dim E)
   │    │     │     │
   ▼    ▼     ▼     │
 ┌──────────────┐   │
 │  SSM Scan    │   │  sequential recurrence over seq_len
 │  (see below) │   │
 └──────┬───────┘   │
        ▼           │
        y           │
        │           │
        ▼           ▼
    ┌─────────────────┐
    │  y * SiLU(z)    │   gating: multiply SSM output by activated z branch
    └────────┬────────┘
             ▼
    ┌─── out_proj ───┐
    │  (E → D=2560)  │
    └────────┬───────┘
             ▼
      Output: (batch, seq_len, 2560)
```

### Key Projections

| Layer | Shape | Purpose |
|:------|:------|:--------|
| `in_proj` | $(D, 2E)$ | Split into $x$ and $z$ branches (gating) |
| `conv1d` | Depthwise, kernel=4 | Local causal context before SSM |
| `x_proj` | $(E, 2 \cdot d_\text{state} + 1)$ | Produces $B$, $C$, and raw $\Delta t$ |
| `dt_proj` | $(1, E)$ | Per-channel step size with bias |
| `out_proj` | $(E, D)$ | Project back to hidden dimension |

## SSM Recurrence

The core scan computes a discretized linear recurrence at each timestep $t$:

$$h_t = \exp(\Delta t_t \cdot A) \cdot h_{t-1} + \Delta t_t \cdot B_t \cdot x_t$$

$$y_t = C_t \cdot h_t + D \cdot x_t$$

Where:

| Symbol | Shape | Description |
|:-------|:------|:------------|
| $h_t$ | $(B_\text{batch}, E, d_\text{state})$ | Hidden state at time $t$ |
| $A$ | $(E, d_\text{state})$ | Diagonal state matrix (learnable, stored as $\log$) |
| $B_t$ | $(B_\text{batch}, d_\text{state})$ | Input-selective matrix at time $t$ |
| $C_t$ | $(B_\text{batch}, d_\text{state})$ | Output-selective matrix at time $t$ |
| $\Delta t_t$ | $(B_\text{batch}, E)$ | Per-channel step size at time $t$ |
| $D$ | $(E,)$ | Skip connection (initialized to ones) |

### Discretization

The step size $\Delta t$ is computed through a softplus projection with clamping:

$$\Delta t = \text{clip}(\text{softplus}(\text{dt\_proj}(dt_\text{raw}) + \text{dt\_bias}),\ 10^{-4},\ 100)$$

This ensures $\Delta t$ stays in a numerically stable range while remaining input-dependent (selective).

### State Initialization

The $A$ matrix is initialized as a repeated diagonal:

$$A_{\log} = \log\!\Big(\text{repeat}\big(\text{arange}(1,\ d_\text{state}+1)\big)_{\text{reps}=E}\Big)$$

At runtime: $A = -\exp(A_{\log})$, producing diagonals from $-1$ to $-d_\text{state}$. The negative exponentials ensure stable decay of the hidden state over time.

## Memory Properties

### Constant Memory Per Token

Unlike standard attention where the KV cache grows as $O(n)$ with sequence length, the SSM maintains a fixed-size state:

| Component | Size |
|:----------|:-----|
| SSM state | $(B_\text{batch},\ E=7680,\ d_\text{state}=16)$ |
| Conv cache | $(B_\text{batch},\ K{-}1=3,\ E=7680)$ |
| **Total per layer** | **~1.5 MB** (FP16, batch=1) |
| **Total 16 SSM layers** | **~24 MB** |

Compare this with a full KV cache for 16 attention layers at 64K context, which would require several GB.

### No KV Cache

SSM layers return `[conv_cache, ssm_state]` as their cache — small, fixed-size tensors. The model's `_create_caches()` method returns `None` for all SSM layers and `KVCache` objects only for the 8 SWA layers (9–16).

## JIT-Compiled Kernels

Two leaf functions are decorated with `@mx.compile` for fused Metal kernel generation (following the Jamba pattern):

### `_ssm_fma`

```python
@mx.compile
def _ssm_fma(a: mx.array, b: mx.array, c: mx.array) -> mx.array:
    return a * b + c    # dA * h + dB * x_t  (fused multiply-add)
```

This fuses the state update $h_t = dA \cdot h_{t-1} + dB \cdot x_t$ into a single kernel, avoiding intermediate tensor allocation.

### `_compute_dt`

```python
@mx.compile
def _compute_dt(dt: mx.array, dt_bias: mx.array, lo: float, hi: float) -> mx.array:
    return mx.clip(nn.softplus(dt + dt_bias), lo, hi)
```

Fuses the bias addition, softplus activation, and clamping into one kernel.

## Autoregressive Decoding

During incremental (token-by-token) generation, the cache mechanism works as follows:

1. **First call** (prefill, `cache=None`): Process the full prompt, initialize `ssm_state` to zeros, build `conv_cache` from the last $K-1$ positions.
2. **Subsequent calls** (decode, `cache=[conv_cache, ssm_state]`): Concatenate `conv_cache` with the new single token, run one step of the scan using the previous `ssm_state`, return updated caches.

The scan loop iterates over `seq_len` positions — during prefill this is the full prompt length; during decode it's exactly 1.

## Parameters

Per SSM layer parameter count (with $D=2560$, $E=7680$, $d_\text{state}=16$):

| Parameter | Shape | Count |
|:----------|:------|:------|
| `in_proj.weight` | $(2E, D)$ | 39.3M |
| `conv1d.weight` | $(E, 1, 4)$ | 30.7K |
| `conv1d.bias` | $(E,)$ | 7.7K |
| `x_proj.weight` | $(33, E)$ | 253.4K |
| `dt_proj.weight` | $(E, 1)$ | 7.7K |
| `dt_proj.bias` | $(E,)$ | 7.7K |
| `A_log` | $(E, 16)$ | 122.9K |
| `D` | $(E,)$ | 7.7K |
| `out_proj.weight` | $(D, E)$ | 19.7M |
| **Total per SSM layer** | | **~60.2M** |

With 16 SSM-bearing layers (8 pure + 8 with MoE), the SSM accounts for roughly **960M parameters** of the 3.2B total.

---

## See also

- [← Architecture Overview](index.md)
- [Axon-SSM Paper](../papers/axon-ssm.md) — Mathematical foundations and selective scan theory
- [SWA + MoE](swa-moe.md) — Sliding window attention and sparse experts used alongside SSM
- [API — Layers](../api/layers.md) — `AxonSSM` and `AxonSSMBlock` Python API
