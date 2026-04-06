# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- CLI: `evaluate` command for WikiText-103 perplexity benchmarking
- CLI: `port-weights` command for Qwen2.5-3B to Bit-Axon format conversion
- CLI: `pipeline` module for end-to-end ML workflow (SFT → merge → quantize → evaluate → inference → ORPO)
- CLI: `prepare` command for dataset format conversion (alpaca, messages, orpo)
- Evaluation: custom tokenizer support in `WikiTextDataset`
- Tests: public API smoke tests and CLI command tests (evaluate, pipeline, port-weights, prepare)

## [0.1.0] - 2026-04-07

### Added

#### Architecture & Model
- `BitAxonModel` — 24-layer sandwich architecture with three block variants (SSM, SWA+MoE, SSM+MoE)
- `BitAxonConfig` — model configuration dataclass (3.2B params, 32K vocab, 65K max context)
- Axon-SSM (Mamba-style State Space Model) with O(1) memory per token and no KV cache
- Shared-Expert Mixture of Experts (8 experts, top-2 routing, ~1.4B active params/token)
- Sliding Window Attention (4K window) for focused reasoning in middle layers
- RMSNorm layer
- KV cache utilities
- `@mx.compile` optimization on leaf functions

#### Training
- LoRA and DoRA (weight-decomposed LoRA) adapter layers
- QLoRA support with `apply_lora_to_model` tree-walker
- SFT, Alpaca, and ORPO dataset classes with batch collation and data pipeline
- ORPO preference optimization (loss, batch collator, trainer)
- Cross-entropy loss and sequence packer
- `TrainingConfig` dataclass with thermal-aware fields
- Cosine LR scheduler with warmup
- Thermal-aware cooling scheduler (pause/stop at configurable temperature thresholds)
- Adapter merging and model export to safetensors
- Core `Trainer` class with gradient accumulation and checkpointing

#### Inference
- Text generation module with autoregressive decoding
- Temperature, top-k, top-p sampling with optional seed control
- Interactive chat mode with streaming output
- Model loading from local safetensors and HuggingFace Hub

#### Quantization
- NF4 (4-bit NormalFloat) quantization scaffold

#### CLI (`bit-axon`)
- `run` — LLM inference with chat mode and streaming
- `train` — Fine-tuning with thermal-aware QLoRA (LoRA/DoRA)
- `quantize` — Weight quantization to lower bit-width
- `merge` — LoRA/DoRA adapter merging into base model
- `benchmark` — Performance benchmarking across sequence lengths
- `download` — Model download from HuggingFace Hub

#### macOS App
- SwiftUI native chat application with MLX-Swift inference backend
- Real-time token streaming
- Token speed and GPU memory monitoring
- Drag-and-drop fine-tuning interface

#### Tokenizer
- `QwenTokenizerWrapper` promoted to top-level `bit_axon.tokenizer` public API

#### Porting
- Tokenizer vocabulary mapping (Qwen2.5 → Bit-Axon)
- Parameter key enumeration for weight alignment
- Weight mapper with embedding and MoE projection support
- Initialization pipeline with validation
- Weight distribution visualization

#### Evaluation
- Perplexity computation module
- WikiText-103 dataset loader

#### Profiling
- Memory profiling utilities
- Speed and thermal profiling with `ThermalMonitor`
- Benchmark suite with background polling and history

#### Infrastructure
- GitHub Actions CI workflow with Dependabot configuration
- Pre-commit hooks (ruff linting and formatting)
- PyPI publishing workflow
- Comprehensive documentation (README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, SUPPORT, issue/PR templates)

### Changed
- Added `typer` and `rich` as dependencies
- Added `tokenizers` and `huggingface_hub` as core dependencies

### Fixed
- Race condition in `ThermalMonitor` `is_rising` guard check
- Ruff lint errors across the codebase

### Removed
- Deprecated standalone `scripts/` directory (all functionality now available via `bit-axon` CLI)
