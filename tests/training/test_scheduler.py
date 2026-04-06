"""Tests for learning rate scheduler."""

import mlx.core as mx

from bit_axon.training.scheduler import build_lr_schedule


class TestBuildLRSchedule:
    def test_warmup_phase_lr_increases(self):
        """During warmup, LR should monotonically increase from 0 to peak."""
        schedule = build_lr_schedule(learning_rate=1e-4, warmup_steps=100, total_steps=1000)
        lrs = [float(schedule(step)) for step in range(0, 101, 10)]
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1], f"LR decreased during warmup at step {i * 10}"

    def test_cosine_phase_lr_decreases(self):
        """After warmup, LR should generally decrease toward min_lr."""
        schedule = build_lr_schedule(learning_rate=1e-4, warmup_steps=100, total_steps=1000, min_lr=0.0)
        lrs = [float(schedule(step)) for step in range(200, 1000, 100)]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1], f"LR increased during decay at step {200 + i * 100}"

    def test_no_warmup(self):
        """With warmup_steps=0, should use pure cosine decay."""
        schedule = build_lr_schedule(learning_rate=1e-4, warmup_steps=0, total_steps=1000)
        lr_0 = float(schedule(0))
        assert lr_0 > 0, "LR at step 0 should be > 0 with no warmup"

    def test_boundary_continuity(self):
        """LR should be continuous at the warmup boundary."""
        warmup_steps = 100
        schedule = build_lr_schedule(learning_rate=1e-4, warmup_steps=warmup_steps, total_steps=1000)
        lr_before = float(schedule(warmup_steps))
        lr_after = float(schedule(warmup_steps + 1))
        diff = abs(lr_before - lr_after)
        assert diff < 1e-6, f"LR jump at warmup boundary: {diff}"

    def test_end_lr_near_min_lr(self):
        """LR at final step should be close to min_lr."""
        schedule = build_lr_schedule(learning_rate=1e-4, warmup_steps=100, total_steps=1000, min_lr=1e-6)
        lr_end = float(schedule(999))
        assert lr_end < 1e-5, f"Final LR {lr_end} should be close to min_lr 1e-6"

    def test_schedule_returns_mx_array(self):
        """Schedule should return mx.array, not float."""
        schedule = build_lr_schedule(learning_rate=1e-4, warmup_steps=10, total_steps=100)
        result = schedule(5)
        assert isinstance(result, mx.array)
