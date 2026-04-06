"""Implementation of the `bit-axon download` command."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from bit_axon.cli._console import print_success

console = Console()


def download_cmd(
    repo_id: Annotated[str, typer.Argument(help="HuggingFace repository ID")] = "skyoo2003/bit-axon",
    local_dir: Annotated[str | None, typer.Option("--local-dir", "-d", help="Local directory to download to")] = None,
    include: Annotated[list[str] | None, typer.Option("--include", help="Glob patterns to include")] = None,
) -> None:
    """Download model weights from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    kwargs: dict[str, object] = {"repo_id": repo_id}
    if local_dir:
        kwargs["local_dir"] = local_dir
    if include:
        kwargs["allow_patterns"] = include

    with console.status(f"[bold green]Downloading from {repo_id}...", spinner="dots"):
        path = snapshot_download(**kwargs)

    print_success(f"Model downloaded to {path}")
