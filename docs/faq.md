# FAQ

## General

### What is Bit-Axon?

Bit-Axon is a 3.2B-parameter hybrid small language model engine built for Apple Silicon. It combines Mamba-style state space models (Axon-SSM), sliding window attention, and mixture-of-experts into a 24-layer sandwich architecture. The entire training-to-inference pipeline runs on a fanless MacBook Air M4 with 16 GB unified memory — no GPU or cloud required.

### What frameworks does Bit-Axon use?

Bit-Axon is built with **Apple MLX**, not PyTorch. This gives it direct access to Metal GPU acceleration on Apple Silicon. Additional dependencies include NumPy, HuggingFace `tokenizers`, `typer` (CLI), and `rich` (terminal UI).

### Is Bit-Axon available on platforms other than macOS?

No. Bit-Axon requires Apple Silicon (M1 or later) and macOS 13+. The MLX framework and thermal monitoring (`powermetrics`) are macOS-specific.

---

## Installation

### MLX installation fails

MLX is installed automatically as a dependency of `bit-axon` via pip. If you encounter issues:

```bash
pip install --upgrade pip
pip install bit-axon
```

If MLX fails to build from source, ensure you have Xcode Command Line Tools:

```bash
xcode-select --install
```

### Python version requirements

Bit-Axon requires **Python 3.10 or later**. It is tested against Python 3.10, 3.11, 3.12, and 3.13.

### "No module named mlx" error

This means MLX was not installed correctly. Try reinstalling:

```bash
pip uninstall mlx -y
pip install bit-axon
```

---

## Training

### Thermal throttling during training

Bit-Axon includes a thermal-aware cooling scheduler that monitors SoC temperature via macOS `powermetrics`. By default:

- **Pause** training at **85°C** (configurable via `--temp-pause`)
- **Stop** training at **95°C** (configurable via `--temp-stop`)

If training pauses frequently:

1. Close other resource-intensive applications
2. Ensure proper ventilation (don't block MacBook air vents)
3. Reduce batch size: `--batch-size 1 --grad-accum-steps 8`
4. Disable thermal monitoring with `--no-thermal` (not recommended for extended sessions)

!!! warning
Disabling thermal monitoring (`--no-thermal`) on a fanless MacBook may cause sustained high temperatures. Monitor manually.

### Out of memory during training

Bit-Axon is designed for 16 GB unified memory. If you encounter OOM:

- Reduce `--max-seq-len` (default 2048)
- Reduce `--batch-size` to 1
- Ensure Q4 quantization is active (default in training pipeline)
- Reduce `--lora-rank` (default 8)

### How do I resume training from a checkpoint?

```bash
bit-axon train data.json --model-weights ./model --resume
```

The `--resume` flag loads the latest checkpoint from the output directory.

---

## Inference

### How do I download the model?

```bash
bit-axon download skyoo2003/bit-axon
```

Or specify a local directory:

```bash
bit-axon download skyoo2003/bit-axon --local-dir ./models/bit-axon
```

### How do I use chat mode?

```bash
bit-axon run --chat
```

Type your message and press Enter. Type `exit` or Ctrl+C to quit.

### Inference is slow

- First inference call may be slower due to MLX compilation caching
- Reduce `--max-tokens` for shorter responses
- Try `--temperature 0` for greedy decoding (slightly faster)

---

## Quantization

### What quantization does Bit-Axon support?

Bit-Axon supports **NF4** (4-bit NormalFloat) quantization with configurable group size (default 64). This reduces model weights from ~6.4 GB (FP16) to ~1.76 GB (Q4).

### How do I quantize a model?

```bash
bit-axon quantize ./model --output ./model-q4 --bits 4 --group-size 64
```

### How do I merge LoRA adapters and re-quantize?

```bash
bit-axon merge ./base-model --adapter ./adapter.safetensors --output ./merged
```

The merge command automatically dequantizes, merges adapters, and re-quantizes.

---

## Getting Help

- **GitHub Issues**: [skyoo2003/bit-axon](https://github.com/skyoo2003/bit-axon/issues)
- **PyPI**: [bit-axon](https://pypi.org/project/bit-axon/)
- **HuggingFace**: [skyoo2003/bit-axon](https://huggingface.co/skyoo2003/bit-axon)

---

## See also

- [Installation](getting-started/installation.md) — Prerequisites and setup guide
- [Training Guide](guides/training.md) — Thermal-aware QLoRA fine-tuning
- [Quantization Guide](guides/quantization.md) — NF4 quantization and memory budgets
- [Inference Guide](guides/inference.md) — CLI and Python API for generation
