"""Shared Rich console helpers for the Bit-Axon CLI."""

from rich.console import Console

console = Console()


def print_success(msg: str) -> None:
    console.print(f"[green]✓[/green] {msg}")


def print_error(msg: str) -> None:
    console.print(f"[red]Error:[/red] {msg}")


def print_warning(msg: str) -> None:
    console.print(f"[yellow]Warning:[/yellow] {msg}")


def print_info(msg: str) -> None:
    console.print(f"[dim]{msg}[/dim]")
