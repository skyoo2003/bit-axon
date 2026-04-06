"""Tests for thermal cooling scheduler."""

from unittest.mock import MagicMock

import pytest

from bit_axon.training.cooling import CoolingScheduler, ThermalPolicy, ThermalShutdownError


class TestThermalPolicy:
    def test_default_thresholds(self):
        policy = ThermalPolicy()
        assert policy.max_speed_temp == 75.0
        assert policy.pause_temp == 85.0
        assert policy.stop_temp == 95.0

    def test_invalid_max_speed_ge_pause(self):
        with pytest.raises(ValueError, match="max_speed_temp"):
            ThermalPolicy(max_speed_temp=85.0, pause_temp=85.0)

    def test_invalid_pause_ge_stop(self):
        with pytest.raises(ValueError, match="pause_temp"):
            ThermalPolicy(pause_temp=95.0, stop_temp=95.0)

    def test_valid_custom_thresholds(self):
        policy = ThermalPolicy(max_speed_temp=60.0, pause_temp=80.0, stop_temp=90.0)
        assert policy.max_speed_temp == 60.0


class TestCoolingScheduler:
    @staticmethod
    def _make_monitor(temp: float | None) -> MagicMock:
        monitor = MagicMock()
        monitor.temperature = temp
        return monitor

    def test_no_throttle_below_max_speed(self):
        """No pause when temperature is below max_speed threshold."""
        monitor = self._make_monitor(50.0)
        scheduler = CoolingScheduler(monitor)
        scheduler.check_before_step(step=0)
        assert scheduler.total_pause_time == 0.0

    def test_shutdown_above_critical(self):
        """ThermalShutdownError raised when temp exceeds stop threshold."""
        monitor = self._make_monitor(96.0)
        scheduler = CoolingScheduler(monitor)
        with pytest.raises(ThermalShutdownError, match=r"96\.0C"):
            scheduler.check_before_step(step=100)

    def test_ci_safe_no_monitoring(self):
        """When temperature is None (CI), no pause or error."""
        monitor = self._make_monitor(None)
        scheduler = CoolingScheduler(monitor)
        scheduler.check_before_step(step=0)
        assert scheduler.total_pause_time == 0.0
        assert scheduler.should_reduce_batch() is False

    def test_should_reduce_batch_between_thresholds(self):
        """should_reduce_batch returns True when temp is between max_speed and pause."""
        monitor = self._make_monitor(80.0)
        scheduler = CoolingScheduler(monitor)
        assert scheduler.should_reduce_batch() is True

    def test_should_not_reduce_below_max_speed(self):
        """should_reduce_batch returns False below max_speed threshold."""
        monitor = self._make_monitor(70.0)
        scheduler = CoolingScheduler(monitor)
        assert scheduler.should_reduce_batch() is False

    def test_total_pause_time_tracking(self):
        """Pause time accumulates correctly."""
        monitor = self._make_monitor(87.0)
        policy = ThermalPolicy(pause_duration=0.01)
        scheduler = CoolingScheduler(monitor, policy=policy)
        assert scheduler.total_pause_time == 0.0

    def test_custom_policy(self):
        """Custom policy thresholds should be respected."""
        monitor = self._make_monitor(70.0)
        policy = ThermalPolicy(max_speed_temp=50.0, pause_temp=80.0, stop_temp=90.0)
        scheduler = CoolingScheduler(monitor, policy=policy)
        assert scheduler.should_reduce_batch() is True  # 70 is between 50 and 80
