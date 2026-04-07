# TurboQuant: KV Cache Compression

**Status**: :fontawesome-solid-clock:{ . amber } Planned
**Source**: [`src/bit_axon/quantization/turboquant.py`](https://github.com/skyoo2003/bit-axon/blob/main/src/bit_axon/quantization/turboquant.py) *(stub)*

## Abstract

TurboQuant is a planned KV cache compression technique for reducing the memory footprint of long-context inference in Bit-Axon. As context lengths grow to the target 64K tokens, the KV cache for the 8 sliding-window attention layers becomes the dominant memory consumer. TurboQuant aims to compress cached key and value tensors to lower precision with minimal quality loss, enabling the full 64K context to fit within the memory budget of a 16 GB MacBook Air.

!!! warning "Planned Feature"

    TurboQuant is referenced from ICLR 2026 submissions and is not yet implemented. The source file currently contains a stub. The details below describe the planned design.

## Key Contributions (Planned)

1. **KV cache quantization** — Compress cached $\mathbf{K}$ and $\mathbf{V}$ tensors from FP16 to 4-bit representations.
2. **Integration with SWA layers** — Applied selectively to the 8 sliding-window attention layers (Zone 2) where KV caches are maintained.
3. **Memory target** — Reduce total inference memory for 64K context from ~2,900 MB to under 2,500 MB.

## Mathematical Foundations

### KV Cache Memory Model

For sliding-window attention with window size $W$, the KV cache per layer requires:

$$
M_{\text{KV}} = 2 \times B \times W \times d_{\text{model}} \times \text{sizeof}(\text{dtype})
$$

where the factor 2 accounts for separate $\mathbf{K}$ and $\mathbf{V}$ tensors. For Bit-Axon's Zone 2 layers:

- $W = 4096$, $d_{\text{model}} = 2560$, $B = 1$ (single batch)
- 8 layers with KV caches
- FP16 (2 bytes per element):

$$
M_{\text{KV}}^{\text{FP16}} = 8 \times 2 \times 4096 \times 2560 \times 2 = 335.5 \text{ MB}
$$

### Quantized KV Cache

TurboQuant targets 4-bit quantization of the KV cache. The compression ratio is:

$$
r = \frac{\text{sizeof}(\text{FP16})}{\text{sizeof}(\text{Q4})} = \frac{16}{4} = 4\times
$$

The quantized KV cache memory:

$$
M_{\text{KV}}^{\text{Q4}} = \frac{M_{\text{KV}}^{\text{FP16}}}{4} = 83.9 \text{ MB}
$$

### Quantization Function

The planned quantization maps FP16 values to 4-bit indices:

$$
\mathbf{K}_{\text{quantized}} = Q(\mathbf{K}) = \text{argmin}_{\mathbf{K}' \in \mathcal{C}_{4\text{-bit}}} \|\mathbf{K} - \mathbf{K}'\|_2
$$

where $\mathcal{C}_{4\text{-bit}}$ is the set of representable values in the 4-bit format. Dequantization reconstructs an approximation:

$$
\hat{\mathbf{K}} = DQ(\mathbf{K}_{\text{quantized}}) \approx \mathbf{K}
$$

### Attention Quality Under Quantization

The attention computation with quantized KV:

$$
\text{Attn}(\mathbf{Q}, \hat{\mathbf{K}}, \hat{\mathbf{V}}) = \text{softmax}\left(\frac{\mathbf{Q}\hat{\mathbf{K}}^T}{\sqrt{d_h}}\right)\hat{\mathbf{V}}
$$

The quality loss is bounded by the quantization error:

$$
\|\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) - \text{Attn}(\mathbf{Q}, \hat{\mathbf{K}}, \hat{\mathbf{V}})\| \leq f(\|\mathbf{K} - \hat{\mathbf{K}}\|, \|\mathbf{V} - \hat{\mathbf{V}}\|)
$$

The specific quantization scheme (NF4, uniform, or learned) is to be determined during implementation.

## Implementation Plan

### Integration Points

| Component | Integration |
|-----------|------------|
| `SlidingWindowAttention` | Replace FP16 KV cache with quantized cache |
| `KVCache` | Add quantize/dequantize methods |
| `turboquant.py` | Core quantization primitives |

### Planned API

```python
# Planned (not yet implemented)
from bit_axon.quantization.turboquant import TurboQuant

quantizer = TurboQuant(bits=4)
# During inference:
# quantizer.compress(kv_cache)  # Compress after each attention step
# quantizer.decompress(kv_cache)  # Decompress for attention computation
```

### Memory Budget Impact

| Configuration | KV Cache Memory | Total Inference Memory |
|--------------|----------------|----------------------|
| FP16, 4K context | 335.5 MB | ~2,500 MB |
| FP16, 64K context | N/A (exceeds window) | ~2,900 MB |
| TurboQuant Q4, 64K context | ~83.9 MB | ~2,500 MB (target) |

## References

- Dettmers, T., et al. (2024). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023. (Related: NF4 quantization.)
- Kwon, W., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023. (Related: KV cache management.)
- Liu, Z., et al. (2024). *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*. arXiv:2402.02750. (Related: KV cache quantization.)
