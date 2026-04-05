"""Tests for thermal monitoring."""

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
