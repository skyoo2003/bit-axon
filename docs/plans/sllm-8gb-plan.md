# Bit-Axon 3B: Master Plan

**"Minimal Bits, Maximal Impulse."**
_A Scalable sLLM Engine for Linear, Sparse, and Quantized Architectures._

> **Document Version**: v3.0 (Consolidated Master Plan)
> **Target Device**: MacBook Air M4 (16GB Unified Memory, ~8GB available for model)
> **Total Duration**: ~6 months (5 phases)

---

## 1. Project Vision

Bit (minimal bits) meets Axon (maximal impulse).

- **Bit (Minimal Bits):** Extreme quantization to minimize memory bandwidth constraints. Weights compressed to 4-bit or 1.58-bit for comfortable operation on 16GB unified memory.
- **Axon (Maximal Impulse):** Fast and efficient signal transmission like human neural axons. Linear and sparse operations deliver maximum inference/training speed with minimal power/heat.
- **Target Environment:** **Thermal-free local training and ultra-fast inference** on a fanless MacBook Air M4.

**3 Core Architectures:**
1. **Linear Architecture**: Mamba-3 based SSM (State Space Model). Memory stays at $O(N)$ regardless of context length. KV-Cache eliminated.
2. **Sparse Architecture**: Shared-Expert MoE. Only ~1.4B of 3.2B total parameters active per token. 60%+ in sleep state.
3. **Quantized Architecture**: Bit-DoRA (training) + NF4/1.58-bit (inference) + TurboQuant (KV cache compression). Breaking memory limits.

**5-Phase Roadmap (6 months):**
- Phase 1: Core Primitives (Week 1-4)
- Phase 2: Architecture Synthesis (Week 5-8)
- Phase 3: Thermal-Aware Training (Week 9-14)
- Phase 4: Alignment & Merging (Week 15-18)
- Phase 5: App & CLI Release (Week 19-24)

**Expected Effects:**
1. **"No GPU, No Cloud"**: Full training-inference-deployment cycle on a fanless MacBook Air, no expensive Nvidia GPU or cloud server needed.
2. **Green AI**: Sparse + Quantized combination drastically reduces power consumption, enabling long LLM sessions on battery.
3. **Scalability**: Starts on M4 Air as minimum unit, scales linearly to Mac Studio (M4 Ultra) and beyond.

---

## 2. Architecture Design

### 2-1. Model Overview

| Property | Value | Notes |
|:---------|:------|:------|
| Total parameters | ~3.2B | 24-layer hybrid |
| Active params/token | ~1.4B | MoE top-2 + 1 shared |
| Layers | 24 | Sandwich structure |
| Hidden dim | 2,560 | d_source_model=2048 (Qwen) |
| MoE experts | 8 | top-2 + 1 shared expert |
| Context Window | up to 64K | SSM + SWA hybrid |
| FP16 model size | ~6,400 MB | All parameters |
| Q4 model size | ~1,760 MB | MLX affine, g=64 |

MoE sparsity keeps per-token active parameters at ~1.4B, so FP16 active weights are only ~2.8GB, fitting well within the 8GB constraint. Inference at Q4 uses ~1.76GB, training with QLoRA uses ~3.2-3.7GB.

---

### 2-2. The 4 Pillars

#### 2-2-1. Linear Module: Axon-SSM (State Space Model)

Replaces standard Transformer self-attention with Mamba-3/Griffin linear RNN structure. Past information is compressed into a fixed-size hidden state, eliminating the need for KV-Cache. Memory growth is $O(1)$ regardless of context length.

#### 2-2-2. Sparse Module: Shared-Expert MoE

Splits the MLP layer into 8 expert networks. Combines an always-active Shared Expert (general knowledge, ~0.2B) with top-2 Routed Experts (~0.4B) selected by routing. Roughly 60% of physical parameters remain in sleep state, keeping chip temperature low.

#### 2-2-3. Quantized Module: Bit-DoRA

LLM speed depends on memory bandwidth more than compute power. For inference, weights are compressed to NF4 (4-bit NormalFloat) or 1.58-bit (Ternary, {-1, 0, 1}). For training, DoRA (Weight-Decomposed LoRA) separates weight magnitude and direction for learning. Achieves Full Fine-tuning-level accuracy with LoRA-level memory (1-2GB).

#### 2-2-4. Quantization Module: TurboQuant

Near-optimal vector quantization engine based on Google Research's ICLR 2026 paper. Operates in two stages: Stage 1 is PolarQuant (random rotation + Beta distribution + per-coordinate optimal quantization), Stage 2 is QJL (1-bit residual correction, unbiased estimator). Distortion guaranteed within ~2.7x Shannon lower bound. Data-oblivious and training-free. Compresses SWA KV caches in layers 9-16 for 6x+ memory savings. (Full details in Section 6)

---

### 2-3. 24-Layer Sandwich Structure

The 24 layers are divided into three zones in a sandwich structure. SSM layers have no KV cache (memory growth is $O(1)$), and only SWA layers consume KV cache.

```
Layer  1-8:  ████████████████████ Pure Axon-SSM (Linear, no KV cache)
Layer  9-16: ████████████████████ SWA + MoE (Attention Window + Sparse)
Layer 17-24: ████████████████████ SSM + MoE (Linear + Sparse)
```

| Zone | Layers | Modules | KV Cache | Role |
|:-----|:------:|:--------|:--------:|:-----|
| Foundation | 1-8 | Axon-SSM | None | Context absorption, linear processing |
| Deep Reasoning | 9-16 | SWA + MoE | Yes | Local reasoning, expert routing |
| Final Synthesis | 17-24 | SSM + MoE | None | Output generation, thermal control |

---

### 2-4. MLX Integration

Designed for perfect native integration with Apple MLX Framework. PyTorch implementation would be inefficient on Mac without CUDA optimization.

1. **MLX JIT Compilation:** Computation graphs wrapped with `mx.compile` decorator, cached as unified kernels for M4 NPU/GPU. Model-level compilation (entire forward pass) is used. Layer-level compilation doesn't work due to MLX module reference issues.
2. **Unified Memory Zero-Copy:** Leverages MLX's unified memory (no distinction between RAM and VRAM arrays). Quantized weights loaded once into memory, CPU/GPU/NPU access simultaneously without data copies.

**Caution:** `shapeless=True` is broken for matmul in MLX <= 0.31. Output shape gets cached from the first trace, producing wrong results on subsequent calls. Use shape-dependent compilation with lazy recompilation on input shape changes.

---

### 2-5. Model Config

```python
from dataclasses import dataclass

@dataclass
class BitAxonConfig:
    """Bit-Axon 3B model configuration"""
    vocab_size: int = 32_000
    hidden_dim: int = 2_560          # d_model
    num_layers: int = 24
    num_heads: int = 32              # SWA heads, head_dim=64
    d_source_model: int = 2048       # Qwen2.5-3B bridge dimension

    # Axon-SSM (Mamba-3 style)
    ssm_d_state: int = 16            # state vector dim
    ssm_d_conv: int = 4              # 1D conv kernel
    ssm_expand: int = 3              # expansion ratio

    # Sliding Window Attention (Layer 9-16 only)
    swa_window_size: int = 4_096     # sliding window

    # Shared-Expert MoE
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_intermediate_dim: int = 4_096   # expert FFN dim
    moe_shared_expert: bool = True

    # General
    weight_tying: bool = True        # embedding = output head
    max_seq_len: int = 65_536        # max 64K context
    rms_norm_eps: float = 1e-6
```

### 2-6. Why This Architecture Fits MacBook Air M4

| Limitation (Air M4) | Bit-Axon Solution | Applied Technology |
|:--------------------|:------------------|:------------------|
| No cooling fan (thermal) | Deactivate 60% of computation to minimize chip load | Sparse (Shared MoE) |
| Limited memory capacity | Compress weights, avoid long context caching | Quantized (4-bit) + Linear (SSM) |
| Insufficient memory during SFT training | Train only direction/magnitude, not full weights | Bit-DoRA (PEFT) |

---

## 3. Memory Budget

### 3-1. 8GB Available Memory Analysis

We calculate exactly how much of the 16GB unified memory is available for the model. macOS and Metal behavior means "all 16GB for the model" is not realistic.

| Item | Size | Notes |
|:-----|:---:|:------|
| **Physical RAM** | 16,384 MB | MacBook Air M4 16GB |
| macOS system usage | ~2,500 MB | macOS 15 + services |
| Other apps | ~2,000 MB | Browser, IDE, etc. |
| MLX overhead | ~500 MB | Framework runtime |
| Metal GPU safety margin | ~1,500 MB | Swap prevention buffer |
| **Available for model** | **~8,000 MB** | **This is the constraint** |

> Closing all other apps may yield ~10GB, but ~8GB is the baseline for stable operation.

---

### 3-2. Inference Memory

At FP16, 4K context uses ~7.2GB, which is very tight. Q4 quantization is the default strategy, and TurboQuant further compresses the KV cache for long contexts.

**Memory by precision:**

| Precision | Weights | Notes |
|:---------:|:-------:|:------|
| FP16 | 6,400 MB | 3.2B x 2 bytes |
| Q8 (affine) | ~3,200 MB | |
| Q4 (affine, g=64) | ~1,760 MB | MLX `nn.QuantizedLinear` |
| 1.58-bit (Ternary) | ~640 MB | BitNet, experimental |

**SSM state memory (fixed, Layer 1-8, 17-24):** ~1.2MB

**SWA KV cache (Layer 9-16, 32 heads, head_dim=64, FP16):**

| Context | KV Cache (FP16) | KV Cache (TurboQuant 3-bit) |
|:-------:|:---------------:|:---------------------------:|
| 1,024 | 64 MB | ~11 MB |
| 4,096 | 256 MB | ~43 MB |
| 16,384 | 1,024 MB | ~171 MB |
| 32,768 | 2,048 MB | ~341 MB |
| 65,536 | 4,096 MB | ~683 MB |

> Formula: 8 layers x 2 (K+V) x 32 heads x 64 dim x seq_len x 2 bytes = 65,536 x seq_len bytes

**Total inference memory:**

| Precision | 4K ctx | 16K ctx | 32K ctx | 64K ctx |
|:---------:|:------:|:-------:|:-------:|:-------:|
| FP16 (no TQ) | ~7,169 MB | ~7,925 MB | ~8,949 MB | ~10,901 MB |
| Q4 (no TQ) | ~2,529 MB | ~3,285 MB | ~4,309 MB | ~6,261 MB |
| Q4 + TQ 3-bit | ~2,306 MB | ~2,432 MB | ~2,602 MB | ~2,944 MB |

**Key conclusion**: Q4 quantization alone fits within 8GB up to 32K. TurboQuant is essential at 64K.

---

### 3-3. Training Memory

From-scratch pre-training (FP16 full fine-tuning) requires ~38.4GB, which is physically impossible. QLoRA (Q4 frozen base + LoRA adapters) compresses training memory to ~3.2-3.7GB.

**Full fine-tuning (Not feasible):**

| Component | Size |
|:----------|:----:|
| Weights FP16 | 6,400 MB |
| Gradients FP16 | 6,400 MB |
| Adam optimizer (m, v FP32) | 25,600 MB |
| **Total** | **~38,400 MB** |

**QLoRA training (Recommended):**

| Component | Size | Notes |
|:----------|:----:|:------|
| Base weights Q4 (frozen) | ~1,760 MB | No gradients |
| LoRA params (r=16, ~0.5%) | ~32 MB | Trainable |
| LoRA gradients (FP16) | ~32 MB | |
| LoRA Adam (m, v FP32) | ~256 MB | |
| Activations (checkpointed, B=1, T=2048) | ~500-1,000 MB | |
| SWA KV cache (T=2048) | ~128 MB | |
| MLX overhead | ~500 MB | |
| **Total** | **~3,208-3,708 MB** | Fits within 8GB |

---

## 4. Training Plan

### 4-1. QLoRA Training Strategy

Since from-scratch training is impossible, we use a **dual-track approach**. Port weights from an existing 3B open-source model (e.g., Qwen2.5-3B) to Bit-Axon architecture, then fine-tune with QLoRA for domain-specific adaptation.

| Track | Method | Memory | Purpose |
|:------|:-------|:------:|:--------|
| **Track A: QLoRA SFT** | Q4 frozen + Bit-DoRA | ~3.5 GB | Domain-specific fine-tuning |
| **Track B: ORPO alignment** | Q4 frozen + ORPO | ~3.8 GB | Preference-based alignment |

> ORPO performs SFT and preference alignment simultaneously without a reference model, so it works within 8GB.

---

### 4-2. QLoRA Training Setup

```python
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW
from mlx.nn import value_and_grad

def setup_qlora_training(model_path: str, lora_rank: int = 16):
    """QLoRA training setup (MLX)"""
    # 1. Load model and Q4 quantize
    model = BitAxonModel.load(model_path)
    model.quantize(bits=4, group_size=64)  # MLX native Q4

    # 2. Insert LoRA adapters (on frozen base)
    #    Bit-DoRA approach: separate Magnitude and Direction
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
            module.enable_lora(r=lora_rank, alpha=lora_rank * 2)
            module.enable_dora()  # direction/magnitude separation

    # 3. Optimizer (adapter params only)
    trainable_params = [
        (k, v) for k, v in model.parameters().items()
        if "lora" in k or "dora" in k
    ]
    optimizer = AdamW(
        learning_rate=2e-4,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    return model, optimizer, trainable_params


def train_step(model, batch, optimizer, config):
    """Single QLoRA training step"""
    input_ids, labels = batch

    def loss_fn(model):
        logits = model(input_ids)
        logits = logits[:, :-1, :]
        labels_shifted = labels[:, 1:]
        loss = mx.mean(
            mx.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                labels_shifted.reshape(-1),
            )
        )
        return loss

    loss, grads = value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return loss
```

**Hyperparameters:**

| Parameter | Value | Notes |
|:----------|:-----:|:------|
| LoRA rank | 16 | ~0.5% of model |
| LoRA alpha | 32 | rank x 2 |
| Learning rate | 2e-4 | Cosine annealing |
| Batch size | 1 | B=1 (memory saving) |
| Gradient accumulation | 8 | Effective batch 8 |
| Sequence length | 2,048 | |
| Max steps | ~10K | Domain SFT |
| Gradient clip | 1.0 | max_norm |

---

### 4-3. Thermal-Aware Scheduler

The 3B model performs more computation than a 250M model, making thermal management more important. Track SoC temperature in real time (1-second intervals) via macOS `powermetrics` or IOKit through Python subprocess. Connect temperature data with the MLX training loop. Leverage MoE sparsity to reduce chip load.

| Temp | Action | Impact |
|:----:|:-------|:-------|
| < 75C | Normal speed | None |
| 75-85C | Halve LoRA rank | Same memory, less compute |
| 85-95C | 0.5s pause | Slight delay |
| > 95C | Pause training | Resume from checkpoint |

> **MoE thermal benefit**: With ~1.4B active params per token, only ~56% of the compute of a dense 3B model. High likelihood of stable training on a fanless MacBook Air.

---

## 5. Inference Plan

### 5-1. Quantization Strategy

Q4 quantization is the default. MLX's `nn.QuantizedLinear` uses Metal-optimized kernels, far faster than custom NF4 dequantize+matmul. Use MLX's built-in `nn.quantize()` to replace all `nn.Linear` with `nn.QuantizedLinear`.

```python
def prepare_inference_model(model_path: str, bits: int = 4):
    """Prepare model for inference"""
    model = BitAxonModel.load(model_path)
    model.quantize(bits=bits, group_size=64)

    # Optional: MLX JIT compilation (speed boost)
    # Requires MoE numpy interop patch (replace with pure MLX MoE forward)
    model.compile()

    return model
```

| Method | MLX | Metal | Speed | Use |
|:-------|:---:|:-----:|:-----:|:----|
| FP16 | Native | Works | Baseline | Debugging |
| Q4 affine | Native | Works | ~2-3x faster | **Default inference** |
| 1.58-bit Ternary | Custom | Works | ~4-6x faster | Experimental / optimized |

---

### 5-2. Context Length Analysis

Recommended settings vary by use case. General chat/coding is fine at 4K. Activate TurboQuant when analyzing long documents.

| Scenario | Context | Config | Total Memory |
|:---------|:-------:|:-------|:------------:|
| Chatbot | 4K | Q4 | ~2.5 GB |
| Code gen | 8K | Q4 | ~2.8 GB |
| Doc summary | 16K | Q4 | ~3.3 GB |
| PDF analysis | 32K | Q4 | ~4.3 GB |
| Full 64K | 64K | Q4 + TurboQuant | ~2.9 GB |
| No TQ | 64K | Q4 only | ~6.3 GB |

---

### 5-3. Expected Inference Performance

Estimated based on M4 memory bandwidth ~120 GB/s. Q4 weights (~1.76GB) load in ~15ms.

| Metric | Q4 (MLX) | Q4 + Compile | Notes |
|:-------|:--------:|:------------:|:------|
| Load | ~0.5s | ~0.8s | Includes compile overhead |
| TTFT (4K) | ~200ms | ~100ms | Prefill |
| tok/s | ~40-60 | ~60-80 | Decode, B=1 |
| Memory (4K ctx) | ~2.5 GB | ~2.5 GB | Same |

---

## 6. TurboQuant Integration

### 6-1. Technical Overview

TurboQuant is an online vector quantization algorithm developed by **Google Research** and presented at **ICLR 2026**. It offers a theoretically robust new approach that overcomes the limitations of existing quantization methods (NF4, AWQ, GPTQ, etc.).

| Property | Existing methods (NF4, AWQ, GPTQ) | TurboQuant |
|:---------|:----------------------------------|:-----------|
| **Target** | Model weights (static, one-time) | KV cache (dynamic, real-time) + weights |
| **Data dependency** | Calibration data needed | **Data-oblivious**: no data needed |
| **Memory overhead** | Per-block quantization constants | **Zero overhead**: no constants to store |
| **Training** | Some methods need fine-tuning | **Training-free** |
| **Theory** | Empirical | **Information-theoretic near-optimal** |
| **Timing** | Offline preprocessing | **Online, instant** |

---

### 6-2. Two-Stage Algorithm (PolarQuant + QJL)

#### Stage 1: PolarQuant (High-Quality MSE Compression)

1. **Random Rotation**: Multiply input vector by a random orthogonal matrix
2. **Beta Distribution Induction**: Transform rotated coordinates to follow a concentrated Beta distribution
3. **Per-Coordinate Optimal Quantization**: At high dimensions, coordinates become nearly independent, so apply individually optimal scalar quantization (Lloyd-Max) to each
4. **Polar Coordinate Transform**: Convert Cartesian to radius + angle to eliminate normalization overhead

> Random rotation simplifies the geometric structure of data, fundamentally eliminating the "memory overhead" hidden in existing methods.

#### Stage 2: QJL — 1-bit Residual Correction (Quantized Johnson-Lindenstrauss)

MSE-optimal quantization introduces bias in inner product estimation. To solve this:

1. Apply 1-bit QJL transform to Stage 1's **residual**
2. Compress each residual coordinate to a single **sign bit** ({+1, -1})
3. Use a special **unbiased estimator** to remove inner product error

> Result: Unbiased near-optimal inner product quantizer complete.

---

### 6-3. Theoretical Performance Guarantees

TurboQuant's distortion rate is guaranteed within ~2.7x the Shannon lower bound. Higher vector dimension (d) results in lower inner product distortion.

| Bitwidth (b) | MSE Distortion (D_mse) | Inner Product Distortion (D_prod/d) |
|:-------------|:----------------------:|:------------------------------------:|
| 1-bit | 0.36 | 1.57/d |
| 2-bit | 0.117 | 0.56/d |
| 3-bit | 0.03 | 0.18/d |
| 4-bit | 0.009 | 0.047/d |

---

### 6-4. SWA KV Cache Compression

The biggest value of TurboQuant is **KV cache compression**. In the 24-layer sandwich structure, Key/Value tensors generated by Sliding Window Attention (SWA) in layers 9-16 are compressed in real time.

| Layer | Type | KV Cache | TurboQuant |
|:-----:|:-----|:--------:|:----------:|
| 1-8 | Pure SSM (Axon-SSM) | None | N/A |
| 9-16 | SWA + MoE | Yes | **Apply (primary target)** |
| 17-24 | SSM + MoE | None | N/A |

**Effects by TurboQuant setting:**
- **3-bit setting**: Zero quality loss, KV cache **6x+ memory savings**
- **4-bit setting**: Attention logit computation speed **up to 8x improvement**

**Expected effects at 65,536 tokens:**

| Metric | Baseline FP16 KV | TurboQuant 3-bit | Improvement |
|:-------|:----------------:|:----------------:|:------------|
| KV cache memory | ~4.8 GB | ~0.8 GB | 6x savings |
| 64K context handling | OOM risk | Stable | Memory limit overcome |
| Perplexity change | Baseline | < 0.3 | Near-lossless |

```python
def enable_turboquant_kv(model, bits: int = 3):
    """Enable TurboQuant KV cache compression on SWA layers (9-16)"""
    for i in range(8, 16):  # SWA layers only
        layer = getattr(model, f"layer_{i}")
        if hasattr(layer, "attention"):
            layer.attention.enable_turboquant(bits=bits)
    return model

def disable_turboquant_kv(model):
    """Disable TurboQuant (saves memory for short contexts)"""
    for i in range(8, 16):
        layer = getattr(model, f"layer_{i}")
        if hasattr(layer, "attention"):
            layer.attention.disable_turboquant()
    return model
```

**When TurboQuant matters:**

| Context | Q4 only | Q4 + TurboQuant | Needed? |
|:-------:|:-------:|:----------------:|:-------:|
| 4K | ~2.5 GB | ~2.3 GB | No |
| 16K | ~3.3 GB | ~2.4 GB | No |
| 32K | ~4.3 GB | ~2.6 GB | Recommended |
| 64K | ~6.3 GB | ~2.9 GB | **Essential** |

---

### 6-5. MLX Implementation Plan

```
src/bit_axon/quantization/
├── nf4.py              # Existing NF4 (keep)
├── ternary.py          # Existing BitNet 1.58-bit (keep)
└── turboquant.py       # New: TurboQuant implementation
    ├── polarquant.py       # Stage 1: Random Rotation + scalar quantization
    ├── qjl.py              # Stage 2: 1-bit residual correction
    └── codebooks.py        # Pre-computed optimal codebooks per bitwidth
```

**Implementation considerations:**
- `mx.compile` compatibility: Fix Random Rotation matrix outside compile cache to minimize recompilation
- Metal kernel optimization: Integrate Random Rotation matmul, scalar quantization lookup, and QJL sign ops into Metal kernels
- Unified memory: Load codebook (lookup table) once, share access across CPU/GPU/NPU
- Thermal impact: Measure how TurboQuant operations interact with the Smart Cooling Scheduler

#### Open-Source Reference Implementations

| Repository | Feature | Reference |
|:-----------|:--------|:----------|
| [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx) | Metal fused kernels, 4.6x compression | Metal kernel design, M4 benchmarks |
| [rachittshah/mlx-turboquant](https://github.com/rachittshah/mlx-turboquant) | KV cache specialized | KV cache integration pattern |
| [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) | PyTorch (714 stars) | Accurate algorithm implementation, HuggingFace compatible |

---

### 6-6. Development Roadmap Integration (TQ-T1 ~ TQ-T7)

TurboQuant is developed as a parallel track alongside the existing quantization module. All task statuses are **Pending**.

| Task | Description | Timeline | Status |
|:-----|:------------|:---------|:------:|
| TQ-T1 | Algorithm research and Bit-Axon applicability analysis | Phase 1 (Week 1-4) | Pending |
| TQ-T2 | MLX core implementation (PolarQuant + QJL) | Phase 1-2 boundary (Week 4-6) | Pending |
| TQ-T3 | SWA layer KV cache TurboQuant compression | Phase 3 (Week 9-14) | Pending |
| TQ-T4 | TurboQuant vs NF4 weight quantization benchmark | Phase 2 (Week 5-8) | Pending |
| TQ-T5 | TurboQuant Metal kernel optimization (Apple Silicon) | Phase 3 (Week 9-14) | Pending |
| TQ-T6 | Low-end GPU TurboQuant integration | Phase 8+ (future) | Future |
| TQ-T7 | TurboQuant integration benchmark and documentation | Phase 4 (Week 15-18) | Pending |

#### References

**Papers:**
- **TurboQuant**: Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2026). "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." *ICLR 2026*. [arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874)
- **PolarQuant**: Zandieh, A. et al. (2026). *AISTATS 2026*. [arxiv.org/abs/2502.02617](https://arxiv.org/abs/2502.02617)
- **QJL**: Zandieh, A. et al. (2024). "Quantized Johnson-Lindenstrauss." *AAAI 2025*. [arxiv.org/abs/2406.03482](https://arxiv.org/abs/2406.03482)

**Google Research Blog:**
- [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (2026-03-24)

---

## 7. Development Roadmap

> **All task statuses = Pending.** No work has been completed.

### Phase 1: Core Primitives (Week 1-4)

**Period:** Week 1-4 (1 month)
**Goal:** Implement Bit-Axon's 3 core architectures using Apple MLX Framework low-level APIs

- **Week 1-2: Linear SSM Kernel Optimization**
  - Implement `Axon-SSM` class (Mamba-3 based) in MLX to replace Transformer Attention blocks
  - Write sequence parallel processing logic with MLX's `mlx.core.scan` (cumulative operations)
  - **Status**: Pending

- **Week 3: Sparse MoE Routing Implementation**
  - `Shared-Expert MoE` routing logic: load all parameters into memory, activate only top-2 experts at compute time
  - MLX array masking to prevent GPU bottlenecks
  - **Status**: Pending

- **Week 4: 4-bit Quantization and Bit-DoRA Scaffolding**
  - Implement logic to compress model weights to 4-bit NormalFloat
  - Initialize DoRA (Weight-Decomposed LoRA) magnitude vector and direction matrix separation
  - **Status**: Pending

---

### Phase 2: Architecture Synthesis (Week 5-8)

**Period:** Week 5-8 (2 months)
**Goal:** Convert and benchmark a 3B-parameter open-source model into Bit-Axon hybrid structure

**Target Metrics:** Inference speed 50-60 tokens/sec, memory <= 2.5GB, power <= 5W

#### Milestone 1: Weight Porting (Week 5-6)

- **T1.1: Test Weight Porting Script** [Pending]
  - Test porting script with Qwen2.5-0.5B, verify 24-layer mapping, check tensor shapes
  - Success criteria: Runs without errors, all required tensors generated, shapes match

- **T1.2: Validate Mapping Quality** [Pending]
  - Load ported weights into BitAxonModel, evaluate perplexity on test set
  - Success criteria: Perplexity delta < 20% vs source model, no NaN/Inf

- **T1.3: Refine Mapping for Target Models** [Pending]
  - Analyze Qwen2.5-3B architecture, map RoPE positional embeddings, handle normalization layers
  - Success criteria: Qwen2.5-3B mapping works, RoPE/normalization layers correctly processed

- **T1.4: Create Weight Validation Tools** [Pending]
  - Source vs target weight comparison script, weight distribution visualization, outlier detection

#### Milestone 2: MLX Compilation & Memory Optimization (Week 7)

- **T2.1: Integrate @mx.compile JIT** [Pending]
  - Apply model-level `mx.compile` (layer-level compilation doesn't work)
  - MoE numpy interop must be replaced with pure MLX patch
  - Success criteria: Compiles without errors, 2x+ speedup, compile time < 10s

- **T2.2: Profile Unified Memory Zero-Copy** [Pending]
  - Use `mx.metal.get_active_memory()`, `mx.metal.get_peak_memory()`
  - Identify and remove unnecessary memory copies
  - Success criteria: Zero unnecessary copies, unified memory benefit confirmed

- **T2.3: Optimize Data Transfer** [Pending]
  - Minimize CPU/GPU transfers, use `mx.array` directly without `.numpy()` conversion
  - Success criteria: < 10% of inference time spent on data transfer

- **T2.4: Cache Compilation Artifacts** [Pending]
  - Save compiled kernels to disk, invalidate cache on model architecture changes

#### Milestone 3: Bare-Metal Benchmarking (Week 8)

- **T3.1: Run Full Model Benchmarks** [Pending]
  - Test at 1K, 4K, 16K, 32K, 64K sequence lengths
  - Success criteria: 50-60 tok/s (4K), memory <= 2.5GB, power <= 5W

- **T3.2: Profile KV Cache with TurboQuant** [Pending]
  - Apply TurboQuant to SWA layers (9-16), measure compression ratio and quality
  - Success criteria: KV cache 6x+ compression, perplexity increase < 5%

- **T3.3: Analyze Thermal Behavior** [Pending]
  - Monitor SoC temperature during sustained inference, measure throttling events
  - Success criteria: SoC temp < 85C, no throttling for 10 minutes

- **T3.4: Generate Optimization Report** [Pending]
  - Document benchmark results, compare against target metrics, recommend Phase 3 priorities

---

### Phase 3: Thermal-Aware Training (Week 9-14)

**Period:** Week 9-14 (1.5 months)
**Goal:** Complete a thermal-controlled SFT training pipeline that overcomes fanless limitations

- **Week 9: Thermal Monitoring Daemon Development** [Pending]
  - Call macOS `powermetrics` or IOKit via Python (subprocess)
  - Real-time tracking of SoC temperature and power consumption at 1-second intervals

- **Week 10-11: Smart Cooling Scheduler Integration** [Pending]
  - Connect temperature data with MLX training loop
  - Logic: 0.5s pause above 85C, pause training at 95C, max speed below 75C

- **Week 12-14: Domain-Specific Training (Full SFT)** [Pending]
  - Prepare high-quality Korean dataset (apply packing technique)
  - Apply Bit-DoRA to train only ~1% of total weights
  - Test overnight unattended SFT with MacBook Air on power adapter

**Success criteria:**
- Stable loss decrease (1000+ steps)
- No OOM during training
- SoC temperature maintained < 85C

---

### Phase 4: Alignment & Merging (Week 15-18)

**Period:** Week 15-18 (1 month)
**Goal:** Reduce hallucination and compress into a single file for final deployment

- **Week 15-16: ORPO Application** [Pending]
  - RLHF/DPO requires loading a reference model into memory, high OOM risk on 16GB
  - Port ORPO to MLX, which performs SFT and preference alignment simultaneously without a reference model

- **Week 17-18: Adapter Merging & Final Quantization** [Pending]
  - Merge 4-bit DoRA weights into base model
  - Temporarily upscale to BF16, then downscale back to 4-bit (GGUF/MLX format)
  - Final artifact: `bit-axon-3b-q4.safetensors` (single file)

---

### Phase 5: App & CLI Release (Week 19-24)

**Period:** Week 19-24 (1.5 months)
**Goal:** Evolve beyond terminal scripts to a native macOS application

- **Week 19-20: CLI Tool Development** [Pending]
  - `bit-axon run "Hello?"`, `bit-axon train my_data.json`, `bit-axon quantize` commands
  - Rust or Python-based terminal utility

- **Week 21-23: SwiftUI Native App** [Pending]
  - macOS-exclusive GUI app using Apple `MLX-Swift` library
  - Chat interface (real-time token speed, SoC temperature display)
  - One-click fine-tuning ("Drag and drop your data")

- **Week 24: Final Release** [Pending]
  - Set up GitHub `Project Bit-Axon` repository
  - Upload completed Korean 3B hybrid model to Hugging Face `mlx-community`

---

### Timeline Summary

| Phase | Duration | Cumul | Deliverable |
|:------|:--------:|:-----:|:------------|
| Phase 1: Core Primitives | Week 1-4 | 4 weeks | SSM, MoE, DoRA primitives |
| Phase 2: Architecture Synthesis | Week 5-8 | 8 weeks | Ported model, Q4 quantization, benchmarks |
| Phase 3: Thermal-Aware Training | Week 9-14 | 14 weeks | QLoRA-trained adapters |
| Phase 4: Alignment & Merging | Week 15-18 | 18 weeks | Merged Q4 model |
| Phase 5: App & CLI Release | Week 19-24 | 24 weeks | CLI, app, open-source release |

Estimated: **~6 months total**

---

## 8. Key Technical Findings

> This section documents technical facts and pitfalls discovered during prior development. Treat as **guidelines**, not completion status.

### 8-1. MLX Platform Findings

1. **Layer-level compilation doesn't work.** Wrapping individual layer `forward` methods with `mx.compile` causes module reference issues. MLX traces the function but can't properly resolve `self` attributes at the layer level. **Workaround: Use model-level compilation.**

2. **`shapeless=True` is broken for matmul in MLX <= 0.31.** Output shape gets cached from the first trace, producing wrong results on subsequent calls. **Workaround: Use shape-dependent compilation, allow lazy recompilation on input shape changes.**

3. **MoE numpy interop breaks `mx.compile`.** When `SharedExpertMoE.forward` uses `np.array()`, `np.where()`, etc., it internally calls `mx.eval()` which breaks lazy tracing. **Workaround: Monkey-patch with `_pure_mlx_moe_forward()` that replaces all numpy ops with pure MLX dense dispatch before compilation.**

4. **MLX `nn.Module` automatically registers `mx.array` attributes to the parameter tree.** No separate `nn.Parameter` class needed. All `mx.array` values assigned as module attributes appear in `parameters()`.

5. **MLX `parameters()` returns nested dictionaries.** To flatten, use `mx.tree_flatten(parameters())` for `(key_path, array)` pairs.

6. **MLX's built-in `nn.QuantizedLinear` uses Metal kernels.** Using `nn.quantize()` is much faster than custom NF4 dequantization + matmul. Quantized matmul runs as a single fused Metal kernel.

7. **Python `for` loops are compatible with `mx.compile` tracing.** Loops with fixed bounds (e.g., `n_experts` iteration) are correctly traced as the tracer executes with concrete values.

### 8-2. Known Pitfalls to Avoid

#### Architecture

- **Duplicate parameter key bug:** Using both `setattr(self, f"layer_{i}", layer)` and `self.layers.append(layer)` in `BitAxonModel.__init__` causes `parameters()` to return the same layer twice. **Fix: Remove `self.layers` list, use only `getattr`.**

- **Missing expert intermediate size:** If `ModelConfig` lacks `expert_intermediate_size`, `AxonSSMMoELayer` can't receive the value. Must also propagate to `AxonSWAMoELayer`. **Fix: Add `expert_intermediate_size: int = 4096` to config.**

- **Dimension projection layers:** When porting Qwen2.5-3B (hidden_size=2048) to Bit-Axon (d_model=2560), the dimension difference must be handled. Setting `d_source_model=2048` creates `input_proj` and `output_proj` layers. Embedding and lm_head operate at 2048 dimensions.

#### Weight Porting

- **"Llama-4-Mini" doesn't exist.** Meta released Llama-4 Scout (109B) and Maverick (402B). No small Llama-4 variant exists. Use Qwen2.5-3B as alternative.

- **Qwen2.5-3B → Bit-Axon mapping:** Map the first 24 of Qwen's 36 transformer layers 1:1, discard layers 24-35. SSM parameters are randomly initialized. MoE expert weights: shared expert from Qwen's MLP, expert 0 is a copy, experts 1-7 are perturbed copies. **517 target keys generated total.**

#### MLX Compilation

- **`CompilationConfig`, `CompilationCache`, `CompilationMetrics` classes are unnecessary.** Compilation is a single function call, not a configurable subsystem. MLX handles kernel caching internally.

- **Persistent kernel caching is unnecessary.** MLX handles this itself, so custom cache layers have no value.

### 8-3. Model Dimension Reference

| Parameter | Qwen2.5-3B (Source) | Bit-Axon (Target) |
|:----------|:---------------------|:------------------|
| hidden_size / d_model | 2048 | 2560 |
| num_layers | 36 | 24 (first 24 mapped, rest discarded) |
| intermediate_size | 11008 | 4096 (expert) / 8192 (shared) |
| vocab_size | 151936 | 32000 (projected via d_source_model=2048) |
| attention | GQA, 2 KV heads | SWA, 32 heads, window=4096 |
| n_experts / top_k | N/A (dense MLP) | 8 experts, top-2 + 1 shared |

**Dimension bridge:** `d_source_model=2048` projects to `d_model=2560` through `input_proj` and `output_proj` linear layers. Embedding and lm_head operate at 2048 dimensions.

---

## 9. Future Expansion

> All content in this section is **Future Consideration**, not part of the current plan.

### 9-1. Multi-Backend Strategy

MLX is Apple-only, making universal expansion impossible with it alone. A multi-backend architecture swaps the engine brain to match each environment.

| Target Hardware | Backend | Key Optimization |
|:----------------|:--------|:-----------------|
| MacBook Air (M4) | Apple MLX | Unified Memory Zero-Copy, Smart Cooling |
| Intel/AMD low-end CPU | llama.cpp (GGML) | BitNet (1.58-bit) pure addition ops, AVX-512 SIMD |
| Low-end GPU server (Nvidia) | vLLM / SGLang (PyTorch) | AWQ 3-bit quantization, CUDA Graph, GaLore SFT |

**Bit-Axon architecture suitability for low-end CPUs:**
- **Linear (SSM):** Prevents memory explosion from KV-Cache even on 8GB RAM low-end PCs
- **Sparse (MoE):** CPUs are weak at MatMul but strong at branching (routing). MoE routing and CPU architecture are a perfect match
- **Quantized (1.58-bit Ternary):** Splitting weights into {-1, 0, 1} converts floating-point multiplication to pure addition. Fast inference even on legacy CPUs with AVX2/AVX-512

### 9-2. Cross-Platform Porting

#### Phase 6: Universal Porting (Month 7-8) — Future Plan

- `mlx-to-gguf` converter: Map Bit-Axon hybrid (Mamba+MoE) architecture to GGUF format recognizable by llama.cpp
- CPU inference optimization: Add 1.58-bit (BitNet) addition operation kernels to GGML backend, tune for 30+ tok/s on legacy Intel i5/i7
- Intel OpenVINO and ONNX format conversion scripts

#### Phase 7: Omni-App Development (Month 9) — Future Plan

- Rewrite SwiftUI-based Mac-exclusive app as Tauri (Rust + React/Vue) desktop app (1/10 the memory of Electron)
- Auto hardware detection: Mac M4 → MLX, Intel CPU → llama.cpp/OpenVINO
- Unified GitHub release as "Bit-Axon Omni"

### 9-3. Micro-Adapter SFT for Low-End CPU

- Introduce ultra-small LoRA (VeRA - Vector-based Random Matrix Adaptation) updating **less than 0.01%** of total parameters
- Train user persona/style from 10-50 text examples in minutes using llama.cpp's CPU training capability

### 9-4. Bit-Axon Swarm (Heterogeneous Distributed Clustering)

#### Phase 8: Low-End GPU Training Pipeline (Month 10-11) — Future Plan

- PyTorch Training Script based on Unsloth optimized kernels (GaLore + DoRA)
- FlashInfer-based CUDA kernels that operate without MoE routing bottlenecks on legacy GPUs (Turing, Ampere)
- AWQ and GPTQ format auto-quantization module

**Low-end GPU training techniques:**
- **GaLore (Gradient Low-Rank Projection):** Project gradients to low dimensions for 3B model Full FT-level performance on 8GB VRAM GPUs
- **LISA (Layer-wise Importance Sampled AdamW):** Load and train only 1 layer at a time, swapping in/out. Extreme VRAM savings

#### Phase 9: Bit-Axon Swarm (Month 12) — Future Plan

- Connect MacBook Air M4 (UI/master), Windows laptop (CPU worker), budget desktop (GTX 1660 worker) on the same network
- **Swarm mode division of labor:**
  - Prefill (prompt processing) → GPU server
  - Decode (token generation) + MoE routing → M4 Mac
  - RAG (knowledge retrieval) → CPU laptop
- Overcome ultra-large context processing and 7B+ model operation (impossible on a single device) through federation of low-end devices

---

## 10. Risk Assessment

The 8GB available constraint introduces additional risks beyond the original 16GB plan. Tight FP16 inference and TurboQuant integration complexity are key risks.

### High Risk

| Risk | Prob. | Impact | Mitigation |
|:-----|:-----:|:------:|:-----------|
| FP16 inference exceeds 8GB | High | Medium | Use Q4 as default. FP16 for debugging only |
| TurboQuant MLX implementation complexity | High | High | Reference existing implementations (arozanov/turboquant-mlx). Run without TQ at 4K-16K |
| MoE + mx.compile compatibility | Medium | High | Apply pure-MLX MoE forward patch (remove numpy interop) |
| OOM during QLoRA training | Low | High | Safe settings: B=1, T=2048. Gradient checkpointing required |
| Weight porting model quality degradation | Medium | High | Perplexity validation. Target < 20% delta. Beware of random SSM param init |
| Performance target miss (50-60 tok/s) | Medium | High | Extensive profiling, hotpath optimization, architecture revision review |

### Medium Risk

| Risk | Prob. | Impact | Mitigation |
|:-----|:-----:|:------:|:-----------|
| 64K context memory overflow | Medium | Medium | TurboQuant mandatory. Never attempt 64K without TQ |
| Thermal throttling (3B model) | Medium | Medium | MoE sparsity reduces compute. Smart Cooling Scheduler required |
| 1.58-bit Ternary stability | Medium | Low | Treat as experimental. Keep Q4 as default |
| Compile/recompile overhead | Medium | Medium | Use shape-dependent instead of shapeless. Allow 2x compilation (prefill+decode) |
| MLX compilation errors (SSM ops) | Low | High | Incremental compilation, isolate problematic ops, uncompiled fallback |

### Low Risk

| Risk | Prob. | Impact | Mitigation |
|:-----|:-----:|:------:|:-----------|
| MLX framework bugs | Low | Medium | Use latest stable version, track Apple MLX GitHub issues |
| Qwen2.5-3B weight format changes | Low | Low | Version control porting scripts, support multiple model sources |
