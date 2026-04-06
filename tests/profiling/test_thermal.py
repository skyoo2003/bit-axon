"""Tests for thermal monitoring."""

import time

from bit_axon.profiling.thermal import ThermalMonitor


class TestThermalMonitor:
    def test_returns_none_or_float(self):
        """Should return None when powermetrics not available (CI/no sudo), or a float."""
        monitor = ThermalMonitor()
        temp = monitor.get_soc_temperature()
        assert temp is None or isinstance(temp, float)

    def test_does_not_raise(self):
        """Should never raise an exception."""
        monitor = ThermalMonitor()
        temp = monitor.get_soc_temperature()
        assert temp is None or isinstance(temp, float)

    def test_reasonable_range(self):
        """If a temperature is returned, it should be in a plausible range."""
        monitor = ThermalMonitor()
        temp = monitor.get_soc_temperature()
        if temp is not None:
            assert 0 < temp < 150

    def test_repeated_calls_no_crash(self):
        """Multiple calls should all succeed without raising."""
        monitor = ThermalMonitor()
        for _ in range(3):
            temp = monitor.get_soc_temperature()
            assert temp is None or isinstance(temp, float)


class TestEnhancedThermalMonitor:
    """Tests for background polling and history features."""

    @staticmethod
    def _make_monitor(temp_sequence: list[float | None]) -> ThermalMonitor:
        """Create a ThermalMonitor with a controllable read function."""
        iterator = iter(temp_sequence)

        def read_fn() -> float | None:
            try:
                return next(iterator)
            except StopIteration:
                return None

        return ThermalMonitor(poll_interval=0.01, history_size=10, read_fn=read_fn)

    def test_start_and_stop_no_exception(self):
        """start() and stop() should complete without raising."""
        monitor = self._make_monitor([50.0, 51.0, 52.0])
        monitor.start()
        monitor.stop()

    def test_temperature_property_reads_background_value(self):
        """temperature property should reflect background readings."""
        monitor = self._make_monitor([65.0] * 20)
        monitor.start()
        time.sleep(0.05)
        temp = monitor.temperature
        monitor.stop()
        assert temp is not None and isinstance(temp, float)

    def test_history_buffer_capped_at_maxlen(self):
        """History should not exceed history_size."""
        monitor = self._make_monitor(list(range(20)))
        monitor.start()
        time.sleep(0.25)
        history = monitor.get_history()
        monitor.stop()
        assert len(history) <= 10

    def test_history_stores_only_valid_readings(self):
        """None readings should not appear in history."""
        monitor = self._make_monitor([60.0, None, 62.0, None, 64.0])
        monitor.start()
        time.sleep(0.06)
        history = monitor.get_history()
        monitor.stop()
        assert all(t is not None for t in history)

    def test_is_rising_detects_increasing(self):
        """is_rising should return True for increasing temperatures."""
        monitor = self._make_monitor([50.0, 55.0, 60.0, 65.0, 70.0])
        monitor.start()
        time.sleep(0.06)
        rising = monitor.is_rising(window=3)
        monitor.stop()
        assert rising is True

    def test_is_rising_false_for_decreasing(self):
        """is_rising should return False for decreasing temperatures."""
        monitor = self._make_monitor([70.0, 65.0, 60.0, 55.0, 50.0])
        monitor.start()
        time.sleep(0.06)
        rising = monitor.is_rising(window=3)
        monitor.stop()
        assert rising is False

    def test_is_above_threshold(self):
        """is_above should compare temperature against threshold."""
        monitor = self._make_monitor([90.0] * 20)
        monitor.start()
        time.sleep(0.05)
        above = monitor.is_above(85.0)
        monitor.stop()
        assert above is True

    def test_is_above_false_when_below(self):
        """is_above should return False when below threshold."""
        monitor = self._make_monitor([50.0])
        monitor.start()
        time.sleep(0.02)
        above = monitor.is_above(85.0)
        monitor.stop()
        assert above is False

    def test_stop_is_idempotent(self):
        """Calling stop() multiple times should not raise."""
        monitor = self._make_monitor([50.0])
        monitor.start()
        monitor.stop()
        monitor.stop()

    def test_none_readings_temperature_none(self):
        """If all readings are None, temperature should be None."""
        monitor = self._make_monitor([None, None, None])
        monitor.start()
        time.sleep(0.04)
        temp = monitor.temperature
        monitor.stop()
        assert temp is None

    def test_backward_compat_get_soc_temperature(self):
        """Original get_soc_temperature() should still work as before."""
        monitor = ThermalMonitor()
        temp = monitor.get_soc_temperature()
        assert temp is None or isinstance(temp, float)
