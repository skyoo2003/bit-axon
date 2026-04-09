from __future__ import annotations

from typing import Any


def _require_datasets() -> Any:
    """Lazily import the *datasets* package with a clear error message."""
    try:
        import datasets

        return datasets
    except ImportError:
        msg = "`datasets` package is not installed. Install with: pip install 'bit-axon[evaluation]'"
        raise ImportError(msg) from None
