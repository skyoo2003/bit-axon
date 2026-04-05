from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WeightStats:
    name: str
    shape: tuple
    mean: float
    std: float
    min_val: float
    max_val: float
    outlier_count: int
    sparsity: float  # fraction of near-zero values (|x| < 1e-6)


def compute_weight_stats(weights: dict[str, object]) -> list[WeightStats]:
    """Compute distribution statistics for all parameters."""
    import mlx.core as mx

    stats: list[WeightStats] = []
    for name, array in weights.items():
        arr = array.astype(mx.float32)
        mean = mx.mean(arr).item()
        std = mx.std(arr).item()
        min_val = mx.min(arr).item()
        max_val = mx.max(arr).item()
        outlier_count = int(mx.sum(mx.abs(arr - mean) > 3 * std).item())
        sparsity = mx.sum(mx.abs(arr) < 1e-6).item() / arr.size
        stats.append(
            WeightStats(
                name=name,
                shape=tuple(arr.shape),
                mean=mean,
                std=std,
                min_val=min_val,
                max_val=max_val,
                outlier_count=outlier_count,
                sparsity=sparsity,
            )
        )
    return stats


def format_stats_table(stats: list[WeightStats], max_rows: int = 20) -> str:
    """Format stats as an ASCII table for CLI output. Truncate to max_rows."""
    sorted_stats = sorted(stats, key=lambda s: s.outlier_count, reverse=True)
    truncated = sorted_stats[:max_rows]

    header = f"{'Name':<40} {'Shape':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Outliers':>8} {'Sparsity':>10}"
    separator = "-" * len(header)
    lines = [header, separator]

    for s in truncated:
        shape_str = str(s.shape)
        lines.append(
            f"{s.name:<40} {shape_str:<20} {s.mean:>10.4f} {s.std:>10.4f} {s.min_val:>10.4f} {s.max_val:>10.4f} {s.outlier_count:>8} {s.sparsity:>10.4f}"
        )

    if len(sorted_stats) > max_rows:
        lines.append(f"... and {len(sorted_stats) - max_rows} more rows")

    return "\n".join(lines)


def detect_anomalies(stats: list[WeightStats]) -> list[str]:
    """Detect anomalous parameters. Returns list of warning strings."""
    warnings: list[str] = []
    for s in stats:
        # All zeros
        if s.max_val == 0.0 and s.min_val == 0.0:
            warnings.append(f"WARNING: {s.name} is all zeros")
            continue

        # All NaN
        import math

        if math.isnan(s.mean) or math.isnan(s.std):
            warnings.append(f"WARNING: {s.name} has NaN values")
            continue

        # Extremely high outlier ratio (>10%)
        total = 1
        for dim in s.shape:
            total *= dim
        if total > 0 and s.outlier_count / total > 0.10:
            warnings.append(f"WARNING: {s.name} has high outlier ratio ({s.outlier_count} outliers)")
            continue

        # Extremely sparse (>99%)
        if s.sparsity > 0.99:
            warnings.append(f"WARNING: {s.name} is extremely sparse ({s.sparsity:.2%} zeros)")
            continue

    return warnings
