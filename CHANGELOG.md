# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-04-07

### Added
- CLI tool (`bit-axon`) with 6 commands: run, train, quantize, merge, benchmark, download
- Autoregressive text generation with temperature, top-k, top-p sampling
- Model loading from local safetensors and HuggingFace Hub
- SwiftUI native macOS app with MLX-Swift for inference
- Real-time token speed and GPU memory monitoring
- Drag-and-drop fine-tuning interface
- PyPI publishing workflow

### Changed
- Promoted `QwenTokenizerWrapper` to top-level `bit_axon.tokenizer`
- Added `typer` and `rich` as dependencies

### Deprecated
- Scripts in `scripts/` directory (use `bit-axon` CLI instead)
