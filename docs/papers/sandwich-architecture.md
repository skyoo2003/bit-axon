# 24-Layer Sandwich Architecture

**Status**: :fontawesome-solid-circle-check:{ .green } Implemented
**Source**: [`src/bit_axon/model.py`](https://github.com/skyoo2003/bit-axon/blob/main/src/bit_axon/model.py), [`src/bit_axon/layers/block.py`](https://github.com/skyoo2003/bit-axon/blob/main/src/bit_axon/layers/block.py)

## Abstract

Bit-Axon employs a 24-layer sandwich architecture where the network is divided into three functional zones, each serving a distinct role in the processing pipeline. The first zone uses pure SSM layers for $\mathcal{O}(1)$ memory context absorption. The middle zone combines sliding window attention with mixture-of-experts for focused reasoning over a 4K token window. The final zone drops attention entirely, using SSM with MoE for fast output synthesis. A dimension bridge projects between the source model dimension (2,048) and the internal hidden dimension (2,560) for Qwen2.5 weight compatibility.

## Key Contributions

1. **Functional layer zoning** — Different computational primitives are assigned to different network depths based on their information-processing characteristics.
2. **Attention-free output zone** — The final 8 layers use no attention mechanism, enabling $\mathcal{O}(1)$ memory per token during output generation.
3. **Dimension bridge** — Input and output projections between $d_{\text{source}} = 2048$ and $d_{\text{model}} = 2560$ allow weight porting from Qwen2.5-3B.
4. **Cache heterogeneity** — Only the middle 8 layers maintain KV caches; SSM layers use internal recurrent state, drastically reducing memory during long-context inference.

## Mathematical Foundations

### Layer Assignment Function

The layer type for index $i \in \{0, 1, \ldots, 23\}$ is determined by:

$$
\text{type}(i) = \begin{cases}
\text{SSM} & \text{if } i < \lfloor L/3 \rfloor \\
\text{SWA+MoE} & \text{if } \lfloor L/3 \rfloor \leq i < \lfloor 2L/3 \rfloor \\
\text{SSM+MoE} & \text{otherwise}
\end{cases}
$$

where $L = 24$ is the total number of layers.

### Full Forward Pass

Given input token indices $\mathbf{t} \in \mathbb{Z}^{B \times S}$:

$$
\mathbf{x}_0 = W_{\text{embed}}[\mathbf{t}] \in \mathbb{R}^{B \times S \times d_{\text{source}}}
$$

$$
\mathbf{x}_0' = \mathbf{x}_0 W_{\text{in}} \in \mathbb{R}^{B \times S \times d_{\text{model}}}
$$

For each layer $i$:

$$
\mathbf{x}_{i+1}' = \text{Block}_i(\mathbf{x}_i')
$$

The output projection maps back:

$$
\mathbf{x}_{\text{out}} = \mathbf{x}_{24}' W_{\text{out}} \in \mathbb{R}^{B \times S \times d_{\text{source}}}
$$

$$
\mathbf{o} = \mathbf{x}_{\text{out}} W_{\text{lm\_head}} \in \mathbb{R}^{B \times S \times V}
$$

With weight tying, $W_{\text{lm\_head}} = W_{\text{embed}}^T$.

### Zone 1: Pure SSM (Layers 0–7)

Each block applies RMSNorm followed by Axon-SSM with a residual connection:

$$
\mathbf{x}_{i+1}' = \mathbf{x}_i' + \text{AxonSSM}(\text{RMSNorm}(\mathbf{x}_i'))
$$

**Memory per token**: $\mathcal{O}(d_{\text{model}} \cdot N_{\text{state}}) = \mathcal{O}(2560 \times 16) = \mathcal{O}(1)$ — constant regardless of sequence length.

### Zone 2: SWA + MoE (Layers 8–15)

Each block applies attention, then MoE, each with its own residual:

$$
\mathbf{x}_{i+1}^{(1)} = \mathbf{x}_i' + \text{SWA}(\text{RMSNorm}(\mathbf{x}_i'))
$$

$$
\mathbf{x}_{i+1}' = \mathbf{x}_{i+1}^{(1)} + \text{MoE}(\text{RMSNorm}(\mathbf{x}_{i+1}^{(1)}))
$$

The sliding window attention uses window size $W = 4096$ with $H = 32$ heads of dimension $d_h = 80$:

$$
\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_h}} \odot \mathbf{M}_{\text{SWA}}\right)\mathbf{V}
$$

where $\mathbf{M}_{\text{SWA}}$ is the sliding window mask: $M_{ij} = -\infty$ if $|i - j| > W$.

**Memory per token**: $\mathcal{O}(W \cdot d_{\text{model}})$ for the KV cache (capped at 4K positions).

### Zone 3: SSM + MoE (Layers 16–23)

Each block applies SSM, then MoE, each with residual:

$$
\mathbf{x}_{i+1}^{(1)} = \mathbf{x}_i' + \text{AxonSSM}(\text{RMSNorm}(\mathbf{x}_i'))
$$

$$
\mathbf{x}_{i+1}' = \mathbf{x}_{i+1}^{(1)} + \text{MoE}(\text{RMSNorm}(\mathbf{x}_{i+1}^{(1)}))
$$

**Memory per token**: $\mathcal{O}(1)$ — no attention KV cache, only SSM recurrent state.

### Parameter Budget

| Zone | Layers | Parameters per Layer | Role |
|------|--------|---------------------|------|
| 1 (SSM) | 0–7 | SSM projections + conv | Context absorption |
| 2 (SWA+MoE) | 8–15 | Attention + 8 experts + shared expert | Deep reasoning |
| 3 (SSM+MoE) | 16–23 | SSM + 8 experts + shared expert | Output synthesis |

The MoE uses shared-expert top-2 routing with 8 experts of intermediate dimension 4,096. The shared expert is always active, providing dense capacity alongside the sparse experts.

## Implementation in Bit-Axon

### Block Variants

Three block classes implement the zone types:

| Class | Zone | Source |
|-------|------|--------|
| `AxonSSMBlock` | 1 (SSM) | `layers/block.py` |
| `AxonSWAMoEBlock` | 2 (SWA+MoE) | `layers/block.py` |
| `AxonSSMMoEBlock` | 3 (SSM+MoE) | `layers/block.py` |

### Cache Management

```python
# From model.py — only SWA+MoE layers create KV caches
def _create_caches(self) -> list:
    caches = []
    for i in range(self.config.num_layers):
        if self._get_layer_type(i, self.config.num_layers) == "swa_moe":
            caches.append(KVCache())
        else:
            caches.append(None)
    return caches
```

### Dimension Bridge

The $d_{\text{source}} = 2048$ dimension enables weight porting from Qwen2.5-3B:

| Projection | Shape | Purpose |
|------------|-------|---------|
| `embed_tokens` | $(V, 2048)$ | Token embedding (shared with lm_head) |
| `input_proj` | $(2048, 2560)$ | Source → internal dimension |
| `output_proj` | $(2560, 2048)$ | Internal → source dimension |
| `lm_head` | $(2048, V)$ | Logits (weight-tied with embed) |

## References

- Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
- Fedus, W., Zoph, B., & Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR 23.
- Qwen Team (2024). *Qwen2.5 Technical Report*.
- Beltagy, I., Peters, M. E., & Cohan, A. (2020). *Longformer: The Long-Document Transformer*. arXiv:2004.05150.
