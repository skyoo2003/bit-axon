"""Shared utilities for CLI tests."""

from __future__ import annotations

import re

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text* so assertions match cleanly."""
    return _ANSI_RE.sub("", text)
