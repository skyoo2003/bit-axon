"""Thermal-aware cooling scheduler for training on fanless MacBooks."""

import time
from dataclasses import dataclass


class ThermalShutdownError(Exception):
    """Raised when SoC temperature exceeds safe operating threshold."""


@dataclass
class ThermalPolicy:
    """Temperature thresholds for thermal-aware training control."""

    max_speed_temp: float = 75.0
    pause_temp: float = 85.0
    stop_temp: float = 95.0
    pause_duration: float = 0.5

    def __post_init__(self):
        if self.max_speed_temp >= self.pause_temp:
            raise ValueError(f"max_speed_temp ({self.max_speed_temp}) must be < pause_temp ({self.pause_temp})")
        if self.pause_temp >= self.stop_temp:
            raise ValueError(f"pause_temp ({self.pause_temp}) must be < stop_temp ({self.stop_temp})")


class CoolingScheduler:
    """Thermal-gated training controller.

    Checks temperature before each training step and applies
    throttling (pauses) or shutdown when thresholds are exceeded.
    """

    def __init__(self, monitor, policy: ThermalPolicy | None = None):
        """Initialize CoolingScheduler.

        Args:
            monitor: ThermalMonitor instance with .temperature property.
            policy: Thermal thresholds. Uses defaults if None.
        """
        if not hasattr(monitor, "temperature"):
            raise TypeError("monitor must have a 'temperature' property")
        self._monitor = monitor
        self._policy = policy or ThermalPolicy()
        self._total_pause_time: float = 0.0

    def check_before_step(self, step: int) -> None:
        """Check temperature and pause/shutdown if needed.

        Args:
            step: Current training step (for error reporting).

        Raises:
            ThermalShutdownError: If temperature exceeds stop threshold.
        """
        temp = self._monitor.temperature
        if temp is None:
            return
        if temp >= self._policy.stop_temp:
            raise ThermalShutdownError(f"SoC temperature {temp:.1f}C exceeds stop threshold {self._policy.stop_temp}C at step {step}")
        while temp is not None and temp >= self._policy.pause_temp:
            time.sleep(self._policy.pause_duration)
            self._total_pause_time += self._policy.pause_duration
            temp = self._monitor.temperature
            if temp is None or temp < self._policy.pause_temp:
                break

    @property
    def total_pause_time(self) -> float:
        """Total time spent in thermal pauses (seconds)."""
        return self._total_pause_time

    def should_reduce_batch(self) -> bool:
        """Whether to reduce batch size (temperature between max_speed and pause)."""
        temp = self._monitor.temperature
        if temp is None:
            return False
        return self._policy.max_speed_temp <= temp < self._policy.pause_temp
