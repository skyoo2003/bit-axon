# Bit-Axon

**Minimal Bits, Maximal Impulse**

A 3.2B-parameter hybrid small language model engine built entirely for Apple Silicon. No GPU. No cloud. Full training, inference, and deployment on a fanless MacBook Air M4.

---

## Install

```bash
pip install bit-axon
```

[![PyPI](https://img.shields.io/pypi/v/bit-axon?color=blue&label=pypi)](https://pypi.org/project/bit-axon/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/skyoo2003/bit-axon)

---

## Why Bit-Axon?

Most LLMs assume you have a data center. Bit-Axon assumes you have a MacBook.

Built with **Python + Apple MLX** (not PyTorch), Bit-Axon runs a 24-layer hybrid architecture with Q4 quantization in roughly 1.76GB of weights. It fits in 16GB of RAM, trains without thermal throttling, and deploys as a native macOS app.

!!! tip "Zero infrastructure"
    No CUDA drivers. No cloud billing. No rented GPUs. A single Apple Silicon machine handles the entire lifecycle from data prep to inference.

---

## Features

=== "Architecture"

    :material-brain: &nbsp; **Hybrid Sandwich Design**

    24-layer architecture stacking Axon-SSM, Sliding Window Attention, and Mixture-of-Experts in a single forward pass. Each layer type handles what it does best.

=== "Efficiency"

    :material-memory: &nbsp; **Q4 Quantization**

    ~1.76GB of weights in 4-bit precision. Runs comfortably on 16GB RAM with room for context windows and KV caches.

=== "Thermal Awareness"

    :material-thermometer: &nbsp; **Powermetrics-Guided Training**

    Reads macOS `powermetrics` in real time. Adapts batch sizes and learning rates to stay within thermal envelopes without manual intervention.

=== "Tooling"

    :material-console: &nbsp; **10 CLI Commands**

    Train, evaluate, quantize, export, chat, and more. Every stage of the model lifecycle has a first-class command.

=== "Native App"

    :material-apple: &nbsp; **SwiftUI macOS Application**

    Drop-in desktop app for inference. No terminal required. Model selection, prompt history, generation parameters, all in a native interface.

=== "Open Stack"

    :material-open-source-initiative: &nbsp; **Fully Open Source**

    MIT licensed. PyPI package for pip installs. HuggingFace model hub for weights and datasets. GitHub for everything else.

---

## Quick Start

Install the package and run your first generation:

```bash title="Terminal"
pip install bit-axon
bit-axon run --model skyoo2003/bit-axon --prompt "Explain quantum entanglement in one sentence."
```

Or fire up an interactive chat session:

```bash title="Terminal"
bit-axon chat --model skyoo2003/bit-axon
```

!!! note "Model weights"
    On first run, weights download automatically from HuggingFace. After that, everything runs locally with no network required.

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.11+ |
| ML Framework | Apple MLX |
| Architecture | Axon-SSM + SWA + MoE |
| Parameters | 3.2B |
| Quantization | Q4 (~1.76GB) |
| Desktop App | SwiftUI (macOS) |
| License | MIT |

---

## Links

[:material-github: GitHub](https://github.com/skyoo2003/bit-axon){ .md-button }
[:simple-huggingface: HuggingFace](https://huggingface.co/skyoo2003/bit-axon){ .md-button }
[:material-package-variant: PyPI](https://pypi.org/project/bit-axon/){ .md-button }

---

## Documentation

[:material-book-open-variant: Getting Started](getting-started/index.md){ .md-button }
[:material-cog: CLI Reference](cli/reference.md){ .md-button }
[:material-sitemap: Architecture](architecture/index.md){ .md-button }
[:material-api: API Reference](api/index.md){ .md-button }
[:material-laptop: macOS App](macos-app/index.md){ .md-button }
[:material-school: Guides](guides/training.md){ .md-button }
[:material-file-document: Papers](papers/index.md){ .md-button }
[:material-help-circle: FAQ](faq.md){ .md-button }
[:material-handshake: Contributing](contributing.md){ .md-button }
