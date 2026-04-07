# SWA + MoE: Attention and Sparse Experts

Layers 9–16 combine Sliding Window Attention (SWA) for local reasoning with a Shared-Expert Mixture-of-Experts (MoE) for sparse computation. Layers 17–24 drop attention and pair Axon-SSM with the same MoE design.

## Sliding Window Attention

### Overview

| Property | Value |
|:---------|:------|
| Hidden dim | 2,560 |
| Num heads | 32 |
| Head dim ($d_k$) | 80 ($2560 / 32$) |
| Window size | 4,096 |
| Complexity | $O(n \cdot w)$ instead of $O(n^2)$ |
| Layers | 9–16 only |
| KV cache | External `KVCache` for incremental decoding |

### Algorithm

```
Input x: (batch, seq_len, 2560)
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
  q_proj  k_proj  v_proj     (each 2560 → 2560)
    │       │       │
    ▼       ▼       ▼
  reshape + transpose  →  (batch, 32, seq_len, 80)
    │       │       │
    │       └───┬───┘
    │           │
    │    ┌──────┴──────┐
    │    │ cache update │  (append K, V if decoding)
    │    └──────┬──────┘
    │           │
    ▼           ▼
    Q          K, V
    │           │
    └─────┬─────┘
          ▼
  ┌───────────────┐
  │  scores = QKᵀ │   scaled by 1/√d_k
  │    / √80      │
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │ + sliding     │   causal AND window mask
  │   window mask │
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │   softmax     │
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │   × V         │
  └───────┬───────┘
          ▼
  reshape + transpose  →  (batch, seq_len, 2560)
          │
          ▼
      o_proj (2560 → 2560)
          │
          ▼
      Output: (batch, seq_len, 2560)
```

### Attention Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

Where $M$ is the combined mask applied element-wise to the attention scores.

### Sliding Window Mask

Each position can attend only to previous positions **within the window**:

$$M_{i,j} = \begin{cases} 0 & \text{if } j \leq i \text{ and } i - j < w \\ -\infty & \text{otherwise} \end{cases}$$

The mask combines two conditions:

1. **Causal**: $j \leq i$ — tokens cannot attend to future positions
2. **Window**: $i - j < 4096$ — tokens cannot attend beyond the window

In code:

```python
causal_mask = k_pos[None, :] <= (q_pos[:, None] + causal_offset)
window_mask = (q_pos[:, None] + causal_offset) - k_pos[None, :] < self.window_size
mask = mx.where(causal_mask & window_mask, 0.0, -mx.inf)
```

The `causal_offset` accounts for KV cache length during autoregressive decoding, where `kv_len > seq_len`.

### KV Cache for Incremental Decoding

Only the 8 SWA layers (9–16) produce a KV cache. During prefill, K and V tensors for the full prompt are stored. During decode, each new token's K/V row is appended:

```
Prefill:  K, V shape = (batch, 32, prompt_len, 80)
Decode:   K, V shape = (batch, 32, prompt_len + n_decoded, 80)
```

The cache grows linearly with generated tokens. For long contexts, TurboQuant compresses these KV tensors by 6× or more (see [Memory Budget](memory-budget.md)).

## Shared-Expert MoE

### Overview

| Property | Value |
|:---------|:------|
| Total experts | 8 routed + 1 shared |
| Top-$k$ | 2 |
| Expert FFN dim | 4,096 |
| Active params per token | ~1.4B (~44% of total 3.2B) |
| Layers | 9–16 (with SWA) and 17–24 (with SSM) |

### Architecture

```
Input x: (batch, seq_len, 2560)
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
 ┌─────────┐   ┌─────────────┐
 │  Router  │   │Shared Expert │   always active
 │ (gate)   │   │  (MLP+GLU)  │
 └────┬────┘   └──────┬──────┘
      │               │
      ▼               │
 ┌──────────────┐     │
 │ softmax gate  │     │
 │ → top-2 pick  │     │
 └──────┬───────┘     │
        │             │
        ▼             │
  ┌──────────┐        │
  │ Expert 0 │        │
  │ Expert 3 │  ← selected by routing
  │   ...    │        │
  └────┬─────┘        │
       │              │
       ▼              ▼
  weighted sum    sigmoid gate × shared_out
       │              │
       └──────┬───────┘
              ▼
           y + shared_out
              │
              ▼
      Output: (batch, seq_len, 2560)
```

### Routing

The gate produces expert scores via a linear projection + softmax:

```python
gates = self.gate(x)                    # (batch, seq_len, 8)
gates = mx.softmax(gates, axis=-1)
inds = mx.argpartition(-gates, kth=k-1)[..., :k]   # top-2 indices
scores = mx.take_along_axis(gates, inds, axis=-1)   # top-2 scores
```

The top-2 indices (`inds`) are passed through `stop_gradient` to prevent routing gradient instability. Expert outputs are weighted by their softmax scores and summed.

### SwitchGLU: Expert-Routed SwiGLU

Each routed expert is implemented as a `SwitchGLU` — a SwiGLU MLP that routes tokens to per-expert weight matrices using MLX's `gather_mm`:

```
Input x + expert indices
        │
   ┌────┼────┐
   ▼    ▼    ▼
gate  up   down      (SwitchLinear per expert)
 │    │    │
 ▼    ▼    │
SwiGLU: SiLU(gate) × up
        │
        ▼
     down_proj
        │
        ▼
    Expert output
```

**Gather-sort optimization**: When the number of tokens exceeds 64, tokens are sorted by expert index before `gather_mm`. This groups same-expert tokens contiguously, improving memory access patterns:

```python
if indices.size >= 64:
    x_sorted, idx_flat, inv_order = _gather_sort(x_exp, indices)
    # ... process sorted ...
    y = _scatter_unsort(y_flat, inv_order, shape=(B, L, K))
```

### Shared Expert

The shared expert is a standard SwiGLU MLP (`gate_proj`, `up_proj`, `down_proj`) applied to **all** tokens unconditionally. Its output is gated by a learned sigmoid:

$$y_\text{shared} = \sigma(W_\text{gate} \cdot x) \cdot \text{MLP}(x)$$

This allows the model to dynamically blend shared knowledge (always available) with routed expert knowledge (specialized):

```python
shared_out = self.shared_expert(x)
shared_out = mx.sigmoid(self.shared_expert_gate(x)) * shared_out
```

### Why a Shared Expert?

| Without shared expert | With shared expert |
|:---------------------|:-------------------|
| All knowledge must be routed | General knowledge always available |
| Router errors cause information loss | Shared expert acts as safety net |
| Load balancing is critical | Less sensitive to routing quality |
| Each expert must re-learn common patterns | Shared expert handles common patterns, experts specialize |

## Block Composition

The SWA+MoE block and SSM+MoE block share the same MoE design but differ in their first sub-layer:

### AxonSWAMoEBlock (Layers 9–16)

```
x → RMSNorm → SWA → (+residual) → RMSNorm → MoE → (+residual) → output
                              ↑                    ↑
                          KV cache            no cache
```

### AxonSSMMoEBlock (Layers 17–24)

```
x → RMSNorm → SSM → (+residual) → RMSNorm → MoE → (+residual) → output
                              ↑                    ↑
                        SSM state             no cache
```

Both blocks use pre-norm residual connections with separate `RMSNorm` instances for each sub-layer.

## Sparsity and Thermal Benefits

With top-2 out of 8 experts, only 25% of routed expert parameters are active per token. Combined with the shared expert:

- **Active parameters per token**: ~1.4B (44% of 3.2B total)
- **Dormant parameters**: ~1.8B (56%) — no compute, no memory bandwidth
- **Thermal impact**: Lower chip utilization enables sustained inference/training on a fanless MacBook Air

The `swiglu` activation function is also JIT-compiled with `@mx.compile`:

```python
@mx.compile
def swiglu(x: mx.array, gate: mx.array) -> mx.array:
    return nn.silu(gate) * x
```

[← Back to Architecture](index.md)
