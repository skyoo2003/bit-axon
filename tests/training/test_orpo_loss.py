"""Tests for ORPO preference optimization loss."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from bit_axon.training.orpo_loss import compute_orpo_loss, get_logps, log1mexp, orpo_loss


class TestLog1mexp:
    def test_correctness(self):
        result = log1mexp(mx.array(-5.0))
        expected = -0.006715
        mx.eval(result)
        assert float(mx.abs(result - expected)) < 1e-4

    def test_boundary(self):
        result = log1mexp(mx.array(-0.6931471805599453))
        mx.eval(result)
        assert mx.isfinite(result)

    def test_large_negative(self):
        result = log1mexp(mx.array(-100.0))
        mx.eval(result)
        assert float(mx.abs(result)) < 1e-10

    def test_zero(self):
        result = log1mexp(mx.array(0.0))
        mx.eval(result)
        assert not mx.isfinite(result)
        assert float(result) == float("-inf")


class TestGetLogps:
    def test_shape(self):
        mx.random.seed(0)
        B, T, V = 2, 5, 10
        logits = mx.random.normal(shape=(B, T, V))
        labels = mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]])
        result = get_logps(logits, labels)
        mx.eval(result)
        assert result.shape == (B,)

    def test_masked_only(self):
        mx.random.seed(1)
        B, T, V = 2, 6, 8
        logits = mx.random.normal(shape=(B, T, V))
        labels = mx.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
        mask_first_half = mx.zeros((B, T - 1))
        mask_first_half[:, 3:] = 1.0
        result = get_logps(logits, labels, mask=mask_first_half)
        mx.eval(result)
        assert result.shape == (B,)
        assert mx.all(mx.isfinite(result))

    def test_all_response(self):
        mx.random.seed(2)
        B, T, V = 3, 4, 12
        logits = mx.random.normal(shape=(B, T, V))
        labels = mx.array([[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 8]])
        result = get_logps(logits, labels)
        mx.eval(result)
        assert result.shape == (B,)
        for i in range(B):
            assert float(result[i]) <= 0.0

    def test_ignore_index(self):
        mx.random.seed(3)
        B, T, V = 1, 6, 10
        logits = mx.random.normal(shape=(B, T, V))
        labels = mx.array([[0, 1, 2, -100, -100, 5]])
        result = get_logps(logits, labels)
        mx.eval(result)
        assert result.shape == (B,)
        assert mx.all(mx.isfinite(result))


class TestOrpoLoss:
    def test_shape(self):
        chosen = mx.array([-1.0, -2.0, -3.0, -4.0])
        rejected = mx.array([-2.0, -3.0, -4.0, -5.0])
        result = orpo_loss(chosen, rejected)
        mx.eval(result)
        assert result.shape == ()

    def test_chosen_better(self):
        chosen = mx.array([-0.5, -0.3])
        rejected = mx.array([-3.0, -4.0])
        loss_good = orpo_loss(chosen, rejected, beta=1.0)
        loss_bad = orpo_loss(rejected, chosen, beta=1.0)
        mx.eval(loss_good, loss_bad)
        assert float(loss_good) < float(loss_bad)

    def test_rejected_better(self):
        chosen = mx.array([-4.0, -3.0])
        rejected = mx.array([-0.5, -0.3])
        result = orpo_loss(chosen, rejected, beta=1.0)
        mx.eval(result)
        assert float(result) > 0.5

    def test_equal_logps(self):
        vals = mx.array([-1.0, -2.0])
        result = orpo_loss(vals, vals, beta=1.0)
        mx.eval(result)
        expected = float(mx.log(mx.array(2.0)))
        assert float(mx.abs(result - expected)) < 1e-5

    def test_beta_scaling(self):
        chosen = mx.array([-2.0])
        rejected = mx.array([-0.5])
        loss_low = orpo_loss(chosen, rejected, beta=0.1)
        loss_high = orpo_loss(chosen, rejected, beta=10.0)
        mx.eval(loss_low, loss_high)
        assert float(loss_high) > float(loss_low)


class _TinyModel(nn.Module):
    def __init__(self, V: int, D: int = 8):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.head = nn.Linear(D, V)

    def __call__(self, x: mx.array) -> mx.array:
        return self.head(self.embed(x))


class TestComputeOrpoLoss:
    def test_end_to_end(self):
        mx.random.seed(42)
        B, T, V = 2, 4, 10
        model = _TinyModel(V)
        chosen_ids = mx.random.randint(shape=(B, T), low=0, high=V)
        chosen_labels = mx.random.randint(shape=(B, T), low=0, high=V)
        rejected_ids = mx.random.randint(shape=(B, T), low=0, high=V)
        rejected_labels = mx.random.randint(shape=(B, T), low=0, high=V)
        total_loss, metrics = compute_orpo_loss(model, chosen_ids, chosen_labels, rejected_ids, rejected_labels)
        mx.eval(total_loss, *metrics.values())
        assert total_loss.shape == ()
        assert mx.isfinite(total_loss)

    def test_metrics_keys(self):
        mx.random.seed(43)
        B, T, V = 2, 4, 10
        model = _TinyModel(V)
        chosen_ids = mx.random.randint(shape=(B, T), low=0, high=V)
        chosen_labels = mx.random.randint(shape=(B, T), low=0, high=V)
        rejected_ids = mx.random.randint(shape=(B, T), low=0, high=V)
        rejected_labels = mx.random.randint(shape=(B, T), low=0, high=V)
        _, metrics = compute_orpo_loss(model, chosen_ids, chosen_labels, rejected_ids, rejected_labels)
        mx.eval(*metrics.values())
        expected_keys = {"nll_loss", "orpo_loss", "chosen_logps", "rejected_logps", "reward_margin"}
        assert set(metrics.keys()) == expected_keys

    def test_no_nan(self):
        mx.random.seed(44)
        B, T, V = 2, 4, 10
        model = _TinyModel(V)
        chosen_ids = mx.random.randint(shape=(B, T), low=0, high=V)
        chosen_labels = mx.random.randint(shape=(B, T), low=0, high=V)
        rejected_ids = mx.random.randint(shape=(B, T), low=0, high=V)
        rejected_labels = mx.random.randint(shape=(B, T), low=0, high=V)
        total_loss, metrics = compute_orpo_loss(model, chosen_ids, chosen_labels, rejected_ids, rejected_labels)
        mx.eval(total_loss, *metrics.values())
        assert mx.isfinite(total_loss)
        for key, val in metrics.items():
            assert mx.isfinite(val), f"NaN in metric '{key}'"

    def test_chosen_higher(self):
        mx.random.seed(45)
        V = 10
        D = 8

        class BiasedChosenModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(V, D)
                self.head = nn.Linear(D, V)

            def __call__(self, x: mx.array) -> mx.array:
                h = self.embed(x)
                logits = self.head(h)
                eye = mx.eye(V)
                return logits + 5.0 * eye[x]

        model = BiasedChosenModel()
        B, T = 4, 6
        chosen_ids = mx.random.randint(shape=(B, T), low=0, high=V)
        chosen_labels = mx.random.randint(shape=(B, T), low=0, high=V)
        rejected_ids = mx.random.randint(shape=(B, T), low=0, high=V)
        rejected_labels = mx.random.randint(shape=(B, T), low=0, high=V)
        _, metrics = compute_orpo_loss(model, chosen_ids, chosen_labels, rejected_ids, rejected_labels, beta=5.0)
        mx.eval(*metrics.values())
        margin = float(metrics["reward_margin"])
        assert margin > 0.0, f"Expected positive reward_margin but got {margin}"
