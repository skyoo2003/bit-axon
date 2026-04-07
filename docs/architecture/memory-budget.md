# Memory Budget

Bit-Axon targets the MacBook Air M4 with 16 GB unified memory, of which roughly 8 GB is available for the model. This page breaks down every byte.

## Available Memory

| Item | Size | Notes |
|:-----|:----:|:------|
| **Physical RAM** | 16,384 MB | MacBook Air M4 16 GB |
| macOS system usage | ~2,500 MB | macOS 15 + services |
| Other apps | ~2,000 MB | Browser, IDE, etc. |
| MLX overhead | ~500 MB | Framework runtime |
| Metal GPU safety margin | ~1,500 MB | Swap prevention buffer |
| **Available for model** | **~8,000 MB** | **This is the constraint** |

!!! tip "Maximizing Available Memory"
    Closing all other apps may yield ~10 GB, but ~8 GB is the baseline for stable operation. All design decisions assume this budget.

## Weight Memory by Precision

| Precision | Weight Size | Fits? |
|:----------|:----------:|:-----:|
| FP16 | ~6,400 MB | ❌ No |
| Q8 (affine) | ~3,200 MB | ⚠️ Tight |
| Q4 (affine, group=64) | ~1,760 MB | ✅ Default |
| 1.58-bit (Ternary) | ~640 MB | 🔬 Experimental |

FP16 weights alone consume 6.4 GB — nearly the entire budget before any activations or KV cache. **Q4 quantization is the default strategy.** MLX's `nn.QuantizedLinear` runs 4-bit matmul as a single fused Metal kernel, delivering 2–3× speedup over FP16 while using 3.6× less memory.

## Parameter Count Breakdown

| Component | Count | FP16 Size | Q4 Size |
|:----------|:-----:|:---------:|:-------:|
| `embed_tokens` | 65.5M | 131 MB | 36 MB |
| `input_proj` | 5.2M | 10.5 MB | 2.9 MB |
| Layers 1–8: AxonSSMBlock (×8) | ~482M | 964 MB | 265 MB |
| Layers 9–16: AxonSWAMoEBlock (×8) | ~1,120M | 2,240 MB | 616 MB |
| Layers 17–24: AxonSSMMoEBlock (×8) | ~1,200M | 2,400 MB | 660 MB |
| `output_proj` | 5.2M | 10.5 MB | 2.9 MB |
| `lm_head` (tied with embed) | 0 | 0 MB | 0 MB |
| **Total** | **~3.2B** | **~6,400 MB** | **~1,760 MB** |

!!! note "Weight Tying Savings"
    `embed_tokens.weight` and `lm_head.weight` share the same tensor, saving ~64 MB in FP16 (~18 MB in Q4).

### SSM State Memory (Fixed, Layers 1–8 and 17–24)

The 16 SSM layers maintain a fixed-size hidden state regardless of context length:

- **SSM state**: $(1, 7680, 16)$ per layer × 16 layers = ~3.7 MB
- **Conv cache**: $(1, 3, 7680)$ per layer × 16 layers = ~0.7 MB
- **Total SSM state**: **~4.4 MB** — negligible

### SWA KV Cache (Layers 9–16, 32 heads, head_dim=80)

Only the 8 SWA layers produce a KV cache. Formula:

$$\text{KV Cache} = 8 \text{ layers} \times 2_{(K+V)} \times 32 \text{ heads} \times 80_{d_k} \times \text{seq\_len} \times 2 \text{ bytes}$$

| Context | KV Cache (FP16) | KV Cache (TurboQuant 3-bit) |
|:-------:|:---------------:|:---------------------------:|
| 1,024 | 80 MB | ~14 MB |
| 4,096 | 320 MB | ~53 MB |
| 16,384 | 1,280 MB | ~213 MB |
| 32,768 | 2,560 MB | ~427 MB |
| 65,536 | 5,120 MB | ~853 MB |

!!! warning "64K Context Without TurboQuant"
    At 65,536 tokens, the FP16 KV cache alone is ~5.1 GB — exceeding the entire available budget. **TurboQuant is essential for 64K context.**

## Total Inference Memory

| Precision | 4K ctx | 16K ctx | 32K ctx | 64K ctx |
|:---------:|:------:|:-------:|:-------:|:-------:|
| FP16 (no TQ) | ~7,233 MB | ~8,393 MB | ~9,673 MB | ~12,233 MB |
| Q4 (no TQ) | ~2,593 MB | ~3,353 MB | ~4,633 MB | ~7,193 MB |
| Q4 + TurboQuant 3-bit | ~2,326 MB | ~2,486 MB | ~2,700 MB | ~3,126 MB |

**Key conclusions:**

- ✅ **Q4 alone** fits within 8 GB up to ~16K context
- ✅ **Q4 + TurboQuant** fits within 8 GB at full 64K context with ~5 GB to spare
- ❌ FP16 inference exceeds 8 GB at any context length beyond 4K

## QLoRA Training Memory

Full fine-tuning (FP16 weights + gradients + Adam optimizer) requires ~38.4 GB — physically impossible on 16 GB. QLoRA (Q4 frozen base + LoRA adapters) compresses training to:

| Component | Size | Notes |
|:----------|:----:|:------|
| Base weights Q4 (frozen) | ~1,760 MB | No gradients needed |
| LoRA params (rank=16, ~0.5%) | ~32 MB | Trainable |
| LoRA gradients (FP16) | ~32 MB | |
| LoRA Adam optimizer ($m$, $v$, FP32) | ~256 MB | |
| Activations (checkpointed, $B=1$, $T=2048$) | ~500–1,000 MB | Gradient checkpointing enabled |
| SWA KV cache ($T=2048$) | ~160 MB | |
| MLX overhead | ~500 MB | |
| **Total** | **~3,240–3,740 MB** | **Fits within 8 GB ✅** |

## Thermal Management

The fanless MacBook Air M4 dissipates all heat passively. Bit-Axon uses a thermal-aware training scheduler that monitors SoC temperature via macOS `powermetrics`:

### CoolingScheduler

| Temperature | Action | Rationale |
|:-----------:|:-------|:----------|
| < 75°C | Normal speed | No throttling needed |
| 75–85°C | Halve LoRA rank | Same memory footprint, less compute |
| 85–95°C | 0.5s pause per step | Forced cooling period |
| > 95°C | Pause training | Resume from checkpoint when cool |

### ThermalPolicy

```python
class ThermalPolicy:
    PAUSE_TEMP = 85    # °C — insert micro-pauses
    STOP_TEMP = 95     # °C — halt training entirely
    RESUME_TEMP = 75   # °C — resume normal operation
    CHECK_INTERVAL = 1 # seconds between temperature checks
```

### MoE Thermal Benefit

MoE sparsity directly reduces thermal load:

- Only ~1.4B of 3.2B parameters are active per token (~44%)
- 56% of the chip's compute units remain idle on each forward pass
- This is roughly equivalent to running a 1.4B dense model in terms of heat generation
- High likelihood of sustained training on a fanless MacBook Air

## Context Length Strategy

| Scenario | Context | Recommended Config | Total Memory |
|:---------|:-------:|:-------------------|:------------:|
| Chatbot | 4K | Q4 | ~2.5 GB |
| Code generation | 8K | Q4 | ~2.8 GB |
| Document summary | 16K | Q4 | ~3.3 GB |
| PDF analysis | 32K | Q4 + TurboQuant recommended | ~2.7 GB |
| Full 64K | 64K | Q4 + TurboQuant **required** | ~3.1 GB |

## Weight Tying Savings

```
embed_tokens:  (32000, 2048) = 65.5M params
lm_head:       (32000, 2048) = 65.5M params  ← same tensor
                                ─────────────
Savings:                       65.5M params = ~131 MB (FP16) / ~36 MB (Q4)
```

By tying `embed_tokens.weight = lm_head.weight`, the model avoids storing a duplicate copy of the embedding matrix. This is particularly valuable at Q4 where every ~36 MB counts toward the 8 GB ceiling.

[← Back to Architecture](index.md)
