# Contributing Guide

Thanks for your interest in contributing to Bit-Axon. This is an open-source small language model engine built for Apple Silicon, and we welcome patches, bug fixes, and new features from the community.

Before participating, please read our [Code of Conduct](https://github.com/skyoo2003/bit-axon/blob/main/CODE_OF_CONDUCT.md).

## Prerequisites

- **macOS** on Apple Silicon (M1 or later)
- **Python 3.10+**
- **Apple MLX 0.31.0+** (installs automatically as a dependency)
- **Git**

A recent version of Xcode Command Line Tools is also recommended (`xcode-select --install`).

## Development Setup

```bash
# Clone the repository
git clone https://github.com/skyoo2003/bit-axon.git
cd bit-axon

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify the installation
python -c "import bit_axon; print(bit_axon.__version__)"
```

The `[dev]` extra pulls in `pytest`, `pytest-xdist`, `ruff`, and `pre-commit`. All project configuration lives in `pyproject.toml`.

!!! tip
    Pre-commit hooks run automatically on every commit. They catch trailing whitespace, missing newlines at end of file, YAML/TOML syntax errors, and Ruff lint issues before they reach CI.

## Development Commands

| Command                           | Description                         |
| --------------------------------- | ----------------------------------- |
| `pytest tests/`                   | Run the full test suite             |
| `pytest -n auto tests/`           | Run tests in parallel               |
| `pytest tests/test_model.py`      | Run a single test file              |
| `pytest -k TestAxonSSM`           | Run tests matching a name pattern   |
| `ruff check src/ tests/`          | Lint for errors and style issues    |
| `ruff check src/ tests/ --fix`    | Auto-fix lint issues where possible |
| `ruff format src/ tests/`         | Format code                         |
| `ruff format --check src/ tests/` | Check formatting without writing    |

## Project Structure

```
bit-axon/
├── src/bit_axon/           # Package source
│   ├── config.py           # Model configuration
│   ├── model.py            # Top-level model definition
│   ├── layers/             # Model layers (SSM, block, MoE, norms, attention)
│   ├── quantization/       # Quantization schemes (NF4, ternary, TurboQuant)
│   ├── training/           # Training adapters (LoRA, DoRA)
│   └── utils/              # Utilities (KV cache, helpers)
├── tests/                  # Test suite, mirrors src layout
│   ├── conftest.py         # Shared fixtures
│   ├── test_config.py
│   ├── test_model.py
│   └── ...
└── docs/                   # MkDocs Material documentation
```

## MLX-Specific Conventions

Bit-Axon is built on Apple's MLX framework, not PyTorch. This is a deliberate choice to run natively on Apple Silicon with minimal overhead.

!!! warning
    Do not import or depend on PyTorch anywhere in the codebase. All tensor operations use `mx.array` from MLX.

### Key differences from PyTorch

**Use `__call__` for forward passes, not `forward()`.** MLX modules use `__call__` as the primary entry point. Avoid defining a separate `forward` method.

**Call `mx.eval()` explicitly.** MLX uses lazy evaluation. When you need a concrete value (for instance in tests or when returning to Python), call `mx.eval()` on the result.

**Subclass `nn.Module`.** All model components inherit from `mlx.nn.Module`. Parameters and buffers are registered by assigning `mx.array` attributes directly.

```python
class MyLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = mx.zeros((dim, dim))  # Auto-registered as parameter

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight
```

!!! info
    When writing new layers or model components, always use `mx.array` for all tensor types. This includes parameters, activations, and any intermediate computations. Never reference `torch.Tensor` or `jax.Array`.

## Code Style

We use **Ruff** for both linting and formatting. No other linters or formatters are needed.

### Formatting

Ruff enforces these rules (configured in `pyproject.toml`):

- Line length: **160 characters**
- **Double quotes** for all strings
- Spaces for indentation (no tabs)
- Target Python version: 3.10

### Linting

Enabled rule sets: E, W, F, I, N, UP, B, SIM, C4, DTZ, RUF. A few specific rules are intentionally ignored (E501, B008, N802, N803). First-party imports (`bit_axon`) are sorted to the top.

### Type Hints

Use type hints for all public function signatures and class attributes. Prefer `from __future__ import annotations` for forward references.

### Docstrings

Use **Google-style docstrings** for all public classes and functions:

```python
def compute_attention_scores(query: mx.array, key: mx.array, scale: float) -> mx.array:
    """Compute scaled dot-product attention scores.

    Args:
        query: Query tensor of shape (batch, heads, seq_len, dim).
        key: Key tensor of shape (batch, heads, seq_len, dim).
        scale: Scaling factor applied before softmax.

    Returns:
        Attention weights of shape (batch, heads, seq_len, seq_len).
    """
```

### Import Order

Ruff's `I` rules handle this automatically. First-party (`bit_axon`) imports come before third-party imports.

!!! tip
    Run `ruff check src/ tests/ --fix` and `ruff format src/ tests/` before committing. The pre-commit hooks do this too, but running them locally lets you fix issues faster.

## Testing

Tests live in `tests/` and **mirror the `src/bit_axon/` directory structure**. If you add a new module at `src/bit_axon/layers/my_layer.py`, its tests go in `tests/layers/test_my_layer.py`. Shared fixtures go in `tests/conftest.py`.

```bash
# Run everything
pytest tests/

# Run in parallel (uses all cores)
pytest -n auto tests/

# Run a specific test file
pytest tests/test_model.py

# Run tests matching a name pattern
pytest -k TestAxonSSM
```

!!! tip
    When writing new tests, follow the existing patterns in the test suite. If you add shared fixtures, put them in `tests/conftest.py` so they're available across all test modules.

!!! note
    All test files must be discoverable by pytest. Make sure files are named `test_*.py` and test functions are named `test_*` or are methods inside classes named `Test*`.

## CLI Development

The CLI is built with [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/).

### Adding a new CLI command

1. Create `src/bit_axon/cli/<command>.py` with a function implementing the command logic
2. Register in `src/bit_axon/cli/main.py` using `@app.command()`:

```python
@app.command()
def mycommand(...):
    from bit_axon.cli.mycommand import mycommand_impl
    mycommand_impl(...)
```

3. Add tests in `tests/cli/test_<command>.py` using `typer.testing.CliRunner`
4. All imports must be lazy (inside functions) to avoid loading MLX for `--help`

### CLI conventions

- Use `--config-small` flag for testing without real models
- Use Rich console for output (spinners, progress bars, tables)
- Lazy import all `bit_axon` modules inside command functions

!!! warning
    Never import MLX or `bit_axon` model code at the module level in CLI files. All imports of MLX-dependent code must be lazy (inside function bodies) so that `bit-axon --help` works without MLX installed.

## Documentation Contributions

The documentation is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and lives in the `docs/` directory.

### Previewing locally

```bash
mkdocs serve
```

This starts a local dev server at `http://localhost:8000`. Changes to markdown files are reflected immediately on reload.

### Editing guidelines

- Write in English. Do not write Korean in the source markdown files.
- Use Material admonitions (`!!! tip`, `!!! warning`, `!!! note`, `!!! info`) for callouts.
- Use proper heading hierarchy. Don't skip levels (for example, don't jump from `##` to `####`).
- Add code blocks with language tags (`` ```python ``, `` ```bash ``, etc.).

### Internationalization (Korean translations)

Bit-Axon supports Korean documentation alongside English. The i18n workflow uses a **file suffix convention**:

- English source: `docs/some-page/index.md`
- Korean translation: `docs/some-page/index.ko.md`

To add or update a Korean translation, create or edit the `.ko.md` file alongside the English source. The build system picks up both versions automatically.

!!! tip
    Only edit the English source files directly. Korean translations go in the matching `.ko.md` files. Never mix languages within a single file.

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/). The format:

```
<type>(<scope>): <description>
```

**Types**: `feat`, `fix`, `docs`, `chore`, `test`, `refactor`, `perf`

**Scopes** in use: `layers`, `model`, `training`, `quantization`, `utils`, `ci`

Examples from the project's history:

```
feat(layers): add selective scan wrapper for AxonSSM
feat(model): implement sparse MoE router with top-k gating
feat(quantization): add TurboQuant mixed-precision quantization
fix(training): correct LoRA gradient accumulation for batched inputs
docs: update README with new benchmark results
chore(ci): add parallel test execution to GitHub Actions
```

!!! info
    Pre-commit hooks validate commit message format automatically. If your commit message doesn't match the conventional commits pattern, the hook will reject it with a suggestion.

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`.
2. **Make your changes** following the style guide above.
3. **Write tests** for new functionality. Place test files in `tests/`, mirroring the `src/bit_axon/` structure. Add shared fixtures to `tests/conftest.py`.
4. **Run linting and tests locally** before pushing:

    ```bash
    ruff check src/ tests/ && ruff format --check src/ tests/
    pytest tests/
    ```

5. **Open a PR** with a clear description of the change and motivation.
6. **AI-assisted contributions**: If any part of your PR was generated or significantly assisted by an AI tool, please note this in the PR description. You don't need to disclose which tool or provide prompts, just flag it so reviewers are aware.
7. **Address review feedback** and push updates to the same branch. The PR will be merged once approved.

## Reporting Bugs

Found a bug? Open an issue with a clear title and include:

- A minimal reproducible example
- Your Python version, macOS version, and MLX version
- The expected vs. actual behavior
- Any relevant logs or stack traces

If you're unsure whether something is a bug, open an issue anyway. We'd rather triage it than miss it.

## License

Bit-Axon is released under the [MIT License](https://github.com/skyoo2003/bit-axon/blob/main/LICENSE). By contributing, you agree that your contributions will be licensed under the same terms.
