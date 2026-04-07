# Installation

## Prerequisites

### Hardware

Bit-Axon runs exclusively on Apple Silicon. You need a Mac with an M1, M2, M3, or M4 series chip. Intel-based Macs are not supported.

### Operating system

macOS 13 (Ventura) or later is required.

### Python

Python 3.10 or later. If you don't have it yet, the easiest path is Homebrew:

```bash
brew install python@3.12
```

!!! tip
    If you work with multiple Python versions, consider using `uv` or `pyenv` to manage your environments. Bit-Axon works with any Python from 3.10 through 3.13.

### MLX

You don't need to install MLX separately. It's declared as a dependency in the `bit-axon` package and installs automatically with `pip`.

## Install the package

```bash
pip install bit-axon
```

This pulls in all runtime dependencies: `mlx`, `numpy`, `tokenizers`, `huggingface_hub`, `typer`, and `rich`.

### Verify the installation

```bash
bit-axon --version
```

```bash
bit-axon --help
```

If both commands print output without errors, you're good to go.

## SwiftUI App (optional)

Bit-Axon ships a native macOS chat application built with SwiftUI. To use it, you need:

- Xcode 15 or later
- macOS 14 (Sonoma) or later

```bash
cd BitAxonApp
open BitAxonApp.xcodeproj
```

Build and run from Xcode. The app provides real-time token streaming, GPU memory monitoring, and drag-and-drop fine-tuning.

!!! warning
    The SwiftUI app requires Xcode and macOS 14+. If you only want the CLI and Python API, you can skip this section entirely.

## Development setup

If you plan to contribute or work on the codebase, install the package in editable mode with dev dependencies:

```bash
git clone https://github.com/skyoo2003/bit-axon.git
cd bit-axon
pip install -e ".[dev]"
```

The `[dev]` extra adds `pytest`, `pytest-xdist`, `ruff`, and `pre-commit`.

### Pre-commit hooks

Set up the pre-commit hooks to catch lint issues before you push:

```bash
pre-commit install
```

### Ruff

Ruff handles both linting and formatting. Run it manually with:

```bash
ruff check .
ruff format .
```

The project configuration lives in `pyproject.toml` under `[tool.ruff]`.

## Troubleshooting

**"No module named mlx"**: MLX requires Apple Silicon. If you're on an Intel Mac, this package won't work. If you are on Apple Silicon, try reinstalling with `pip install --force-reinstall mlx`.

**"Python 3.9 or earlier"**: Bit-Axon needs Python 3.10+. Check your version with `python3 --version` and upgrade if needed.

**"Command not found: bit-axon"**: The pip-installed script might not be on your PATH. Try `python3 -m bit_axon` instead, or make sure your pip bin directory is on PATH.

---

## See also

- [Quickstart](quickstart.md) — Download the model and run your first inference
- [FAQ](../faq.md) — Troubleshooting MLX installation and other common issues
