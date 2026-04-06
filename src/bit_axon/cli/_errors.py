"""Exception-to-friendly-message mapping for the CLI."""


def format_error(error: Exception) -> str:
    """Convert an exception to a user-friendly error message."""
    if isinstance(error, FileNotFoundError):
        return f"File not found: {error.filename}"
    if isinstance(error, ImportError):
        return f"Missing dependency: {error.name}. Install with: pip install bit-axon[{get_extra(error)}]"
    if isinstance(error, KeyboardInterrupt):
        return "Interrupted."
    return str(error) if str(error) else type(error).__name__


def get_extra(error: ImportError) -> str:
    msg = str(error).lower()
    if "mlx" in msg:
        return "training"
    return "all"
