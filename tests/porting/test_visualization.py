from __future__ import annotations

import mlx.core as mx
import pytest

from bit_axon.porting.visualization import (
    compute_weight_stats,
    detect_anomalies,
    format_stats_table,
)


def _make_weights():
    return {
        "test.normal": mx.random.normal((100, 50)),
        "test.zeros": mx.zeros((10, 10)),
        "test.sparse": mx.concatenate([mx.zeros((99, 1)), mx.ones((1, 1))], axis=0),
    }


def test_compute_stats_shape():
    weights = {"layer.weight": mx.array([[1.0, 2.0], [3.0, 4.0]])}
    stats = compute_weight_stats(weights)
    assert len(stats) == 1
    s = stats[0]
    assert s.name == "layer.weight"
    assert s.shape == (2, 2)
    assert s.mean == pytest.approx(2.5, abs=1e-4)
    assert s.min_val == pytest.approx(1.0, abs=1e-4)
    assert s.max_val == pytest.approx(4.0, abs=1e-4)
    assert s.sparsity == 0.0


def test_compute_stats_outlier_detection():
    base = mx.zeros((1000,))
    outlier = mx.ones((10,)) * 100.0
    arr = mx.concatenate([base, outlier])
    weights = {"layer.with_outliers": arr}
    stats = compute_weight_stats(weights)
    s = stats[0]
    assert s.outlier_count == 10


def test_format_table_readable():
    stats = compute_weight_stats(_make_weights())
    table = format_stats_table(stats)
    assert isinstance(table, str)
    assert len(table) > 0
    assert "Name" in table
    assert "Shape" in table
    assert "Outliers" in table


def test_detect_anomalies_finds_zeros():
    weights = {"test.zeros": mx.zeros((10, 10))}
    stats = compute_weight_stats(weights)
    warnings = detect_anomalies(stats)
    assert any("all zeros" in w for w in warnings)


def test_detect_anomalies_finds_high_sparsity():
    weights = {"test.sparse": mx.concatenate([mx.zeros((199, 1)), mx.ones((1, 1))], axis=0)}
    stats = compute_weight_stats(weights)
    warnings = detect_anomalies(stats)
    assert any("sparse" in w for w in warnings)
    stats = compute_weight_stats(weights)
    warnings = detect_anomalies(stats)
    assert any("sparse" in w for w in warnings)


def test_compute_stats_no_crash_empty():
    stats = compute_weight_stats({})
    assert stats == []
