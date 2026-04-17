# Bit-Axon

**Minimal Bits, Maximal Impulse**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/bit-axon.svg)](https://pypi.org/project/bit-axon/)
[![Platform: Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-black.svg)]()

Bit-Axon is a 3.2B-parameter hybrid small language model engine built from the ground up for Apple Silicon. It combines Mamba-style state space models, shared-expert mixture-of-experts, and aggressive 4-bit quantization into a single architecture that runs inference on a MacBook Air M4 with 16 GB unified memory. Built with Python and Apple's MLX framework, not PyTorch.

---

## Key Features

- **Linear complexity**: Mamba-style Axon-SSM layers with O(1) memory per token and no KV cache, handling long contexts without the quadratic cost of full attention
- **Sparse activation**: 8-expert shared-expert MoE with top-2 routing, activating only ~1.4B parameters per token while 60% of the model stays idle
- **Aggressive quantization**: NF4 inference, weight-decomposed DoRA fine-tuning, and planned TurboQuant KV cache compression to fit 64K context into under 3 GB

---

## Architecture Overview

Bit-Axon uses a 24-layer sandwich structure where each third of the network serves a distinct role:

```
Layer  1-8:  ████████████████████ Pure Axon-SSM (Linear, no KV cache)    → Context absorption
Layer  9-16: ████████████████████ SWA + MoE (Attention + Sparse)          → Deep reasoning
Layer 17-24: ████████████████████ SSM + MoE (Linear + Sparse)             → Output synthesis
```

The first eight layers are pure SSM, absorbing raw context with constant memory. The middle eight add sliding window attention (4K window) alongside MoE for focused reasoning. The final eight drop attention entirely, relying on SSM plus sparse experts for fast output synthesis.

### Model Configuration

| Parameter              | Value  | Notes                            |
| ---------------------- | ------ | -------------------------------- |
| `vocab_size`           | 32,000 | Tokenizer vocabulary             |
| `hidden_dim`           | 2,560  | Model width (d_model)            |
| `num_layers`           | 24     | Total transformer/SSM layers     |
| `num_heads`            | 32     | SWA attention heads              |
| `head_dim`             | 80     | 2,560 / 32                       |
| `d_source_model`       | 2,048  | Qwen2.5-3B bridge dimension      |
| `ssm_d_state`          | 16     | SSM state vector size            |
| `ssm_d_conv`           | 4      | SSM 1D convolution kernel        |
| `ssm_expand`           | 3      | SSM expansion ratio              |
| `swa_window_size`      | 4,096  | Sliding window attention span    |
| `moe_num_experts`      | 8      | MoE expert count                 |
| `moe_top_k`            | 2      | Active experts per token         |
| `moe_intermediate_dim` | 4,096  | Expert FFN dimension             |
| `moe_shared_expert`    | true   | Shared expert always active      |
| `max_seq_len`          | 65,536 | Maximum context length           |
| `weight_tying`         | true   | Embedding and output head shared |
| `rms_norm_eps`         | 1e-6   | RMSNorm epsilon                  |

---

## Memory Budget

All figures assume a MacBook Air M4 with 16 GB unified memory and roughly 8 GB available for the model.

| Configuration           | Weight Memory | Inference Memory   |
| ----------------------- | ------------- | ------------------ |
| FP16 weights            | ~6,400 MB     | N/A (does not fit) |
| Q4 weights, 4K context  | ~1,760 MB     | ~2,500 MB          |
| Q4 weights, 64K context | ~1,760 MB     | ~2,900 MB          |
| QLoRA training (4-bit)  | ~1,760 MB     | ~3,200–3,700 MB    |

---

## Installation

```bash
pip install bit-axon
```

For development, which pulls in pytest and pytest-xdist:

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+ and an Apple Silicon Mac with MLX installed. The `mlx` and `numpy` dependencies are declared in the package metadata and install automatically.

---

## Quick Start

### CLI

```bash
pip install bit-axon
bit-axon download skyoo2003/bit-axon
bit-axon run "Hello, world!"
bit-axon run --chat  # Interactive chat mode
```

### Python API

```python
import mlx.core as mx
from bit_axon import BitAxonConfig, BitAxonModel

config = BitAxonConfig()
model = BitAxonModel(config)

input_ids = mx.array([[1, 42, 100, 200, 500]])
logits, caches = model(input_ids)

print(f"Output shape: {logits.shape}")  # (1, 5, 32000)
```

The returned `caches` list contains KV cache objects for SWA layers (9 through 16) and `None` for SSM-only layers, since SSM layers maintain internal state without external caching.

---

## CLI Commands

| Command                                                   | Description                              |
| --------------------------------------------------------- | ---------------------------------------- |
| `bit-axon run "prompt"`                                   | Run LLM inference                        |
| `bit-axon train data.json`                                | Fine-tune with SFT (thermal-aware QLoRA) |
| `bit-axon quantize ./model`                               | Quantize model weights                   |
| `bit-axon merge --base-model ./model --adapter ./adapter` | Merge LoRA/DoRA adapters                 |
| `bit-axon benchmark`                                      | Benchmark model performance              |
| `bit-axon download [repo]`                                | Download model from HuggingFace Hub      |

Use `bit-axon <command> --help` for full options.

---

## macOS App

Bit-Axon includes a native SwiftUI app for real-time chat on Apple Silicon.

```bash
cd BitAxonApp
swift build
open BitAxonApp.xcodeproj  # or open in Xcode
```

Features:

- Real-time token streaming
- Token speed and GPU memory monitoring
- Drag-and-drop fine-tuning

---

## Project Structure

```
bit-axon/
├── pyproject.toml              # Build config, dependencies, test config
├── src/bit_axon/
│   ├── __init__.py             # Package version
│   ├── config.py               # BitAxonConfig dataclass
│   ├── model.py                # BitAxonModel (24-layer sandwich)
│   ├── layers/
│   │   ├── axon_ssm.py         # Mamba-style State Space Model
│   │   ├── block.py            # 3 block variants (SSM, SWA+MoE, SSM+MoE)
│   │   ├── moe.py              # Shared-Expert Mixture of Experts
│   │   ├── rms_norm.py         # RMSNorm
│   │   └── swa.py              # Sliding Window Attention
│   ├── quantization/
│   │   ├── nf4.py              # 4-bit NormalFloat quantization
│   │   ├── ternary.py          # 1.58-bit BitNet (planned)
│   │   └── turboquant.py       # TurboQuant KV cache compression (planned)
│   ├── training/
│   │   ├── lora.py             # LoRA adapter
│   │   └── dora.py             # DoRA (weight-decomposed LoRA) adapter
│   └── utils/
│       └── cache.py            # KV cache utilities
├── tests/                      # Mirrors src/bit_axon structure
└── docs/plans/                 # Development plans (EN + KO)
```

---

## Roadmap

1. **Core Primitives** (Weeks 1–4): Axon-SSM, shared-expert MoE, and DoRA adapter implementations
2. **Architecture Synthesis** (Weeks 5–8): Weight porting from Qwen2.5-3B, NF4 quantization, initial benchmarks
3. **Training** (Weeks 9–14): QLoRA supervised fine-tuning with thermal-aware scheduling for sustained training on a fanless MacBook
4. **Alignment** (Weeks 15–18): ORPO preference optimization and adapter merging
5. **Release** (Weeks 19–24): CLI inference tool, SwiftUI chat application, and open-source publication

---

## Links

- **GitHub**: [skyoo2003/bit-axon](https://github.com/skyoo2003/bit-axon)
- **HuggingFace**: [skyoo2003/bit-axon](https://huggingface.co/skyoo2003/bit-axon)
- **PyPI**: [bit-axon](https://pypi.org/project/bit-axon/)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on submitting issues, opening pull requests, and development workflow.

---

## License

Bit-Axon is released under the [Apache License 2.0](LICENSE).
