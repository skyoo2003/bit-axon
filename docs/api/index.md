# API Reference

The Bit-Axon API is organized into the following subpackages.

## Public API

The top-level `bit_axon` package exports the core model classes:

```python
from bit_axon import BitAxonConfig, BitAxonModel, QwenTokenizerWrapper
```

## Subpackages

| Subpackage | Description |
|---|---|
| [`Config`](config.md) | Model configuration dataclass |
| [`Model`](model.md) | 24-layer sandwich model |
| [`Tokenizer`](tokenizer.md) | HuggingFace tokenizer wrapper |
| [`Training`](training.md) | Training pipeline, LoRA, DoRA, ORPO |
| [`Inference`](inference.md) | Autoregressive generation |
| [`Layers`](layers.md) | Core neural network layers |
| [`Quantization`](quantization.md) | NF4 weight quantization |
| [`Profiling`](profiling.md) | Memory, speed, thermal profiling |
| [`Evaluation`](evaluation.md) | Perplexity computation |

---

## See also

- [Architecture Overview](../architecture/index.md) — Model design and configuration details
- [CLI Reference](../cli/reference.md) — Command-line interface for all operations
- [Training Guide](../guides/training.md) — End-to-end fine-tuning walkthrough
- [Inference Guide](../guides/inference.md) — Generation, chat, and streaming
