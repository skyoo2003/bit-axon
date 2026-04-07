# Changelog

All notable changes to the Bit-Axon project will be documented in this file.

## [Unreleased]

### Added
- CLI: `evaluate` command for WikiText-103 perplexity benchmarking
- CLI: `port-weights` command for Qwen2.5-3B to Bit-Axon format conversion
- CLI: `pipeline` module for end-to-end ML workflow (SFT → merge → quantize → evaluate → inference → ORPO)
- CLI: `prepare` command for dataset format conversion (alpaca, messages, orpo)
- Evaluation: custom tokenizer support in `WikiTextDataset`
- Tests: public API smoke tests and CLI command tests

## [0.1.0] - 2026-04-07

### Added

#### Architecture & Model
- `BitAxonModel` — 24-layer sandwich architecture with three block variants
- `BitAxonConfig` — model configuration dataclass (3.2B params, 32K vocab, 65K max context)
- Axon-SSM (Mamba-style State Space Model) with O(1) memory per token
- Shared-Expert MoE (8 experts, top-2 routing, ~1.4B active params/token)
- Sliding Window Attention (4K window)
- RMSNorm layer and KV cache utilities

#### Training
- LoRA and DoRA adapter layers with QLoRA support
- SFT, Alpaca, and ORPO dataset classes
- Thermal-aware cooling scheduler
- Cosine LR scheduler with warmup
- Adapter merging and safetensors export

#### Inference
- Autoregressive text generation with streaming
- Temperature, top-k, top-p sampling
- Interactive chat mode
- Model loading from local path and HuggingFace Hub

#### Quantization
- NF4 (4-bit NormalFloat) quantization

#### CLI
- `run`, `train`, `quantize`, `merge`, `benchmark`, `download` commands

#### macOS App
- SwiftUI native chat application with MLX-Swift backend

#### Infrastructure
- GitHub Actions CI, PyPI publishing, pre-commit hooks
