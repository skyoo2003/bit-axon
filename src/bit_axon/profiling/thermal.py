"""SoC thermal monitoring for Apple Silicon."""

import re
import subprocess
from collections import deque
from collections.abc import Callable
from threading import Event, Lock, Thread


class ThermalMonitor:
    """Read Apple Silicon SoC temperature via powermetrics.

    Supports both single-shot reads (backward compatible) and
    continuous background polling for thermal-aware training.
    """

    def __init__(
        self,
        poll_interval: float = 1.0,
        history_size: int = 60,
        read_fn: Callable[[], float | None] | None = None,
    ):
        """Initialize ThermalMonitor.

        Args:
            poll_interval: Seconds between background polls.
            history_size: Max number of temperature readings to keep.
            read_fn: Optional custom read function (for testing). If None,
                     uses get_soc_temperature internally.
        """
        self._poll_interval = poll_interval
        self._history: deque[float] = deque(maxlen=history_size)
        self._thread: Thread | None = None
        self._stop_event = Event()
        self._lock = Lock()
        self._current_temp: float | None = None
        self._read_fn = read_fn or self.get_soc_temperature

    def get_soc_temperature(self) -> float | None:
        """Get current SoC die temperature in Celsius.

        Uses `sudo powermetrics --samplers smc -i 1 -n 1` on macOS.
        Falls back to None if not available (no sudo, not macOS, etc.).
        """
        try:
            result = subprocess.run(
                ["sudo", "powermetrics", "--samplers", "smc", "-i", "1", "-n", "1"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout + result.stderr
            match = re.search(r"[Dd]ie [Tt]emperature:\s*([\d.]+)\s*C", output)
            if match:
                return float(match.group(1))
            return None
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return None

    def start(self) -> None:
        """Start background temperature polling thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background polling and wait for thread to finish."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self._poll_interval * 2 + 1)
            self._thread = None

    @property
    def temperature(self) -> float | None:
        """Thread-safe access to the most recent temperature reading."""
        with self._lock:
            return self._current_temp

    def get_history(self) -> list[float]:
        """Return a snapshot of the temperature history buffer."""
        with self._lock:
            return list(self._history)

    def is_rising(self, window: int = 5) -> bool:
        """Check if temperature trend is rising over the last N readings.

        Uses simple linear regression slope. Returns False if fewer than
        2 readings are available in the window.
        """
        with self._lock:
            history = list(self._history)
        if window < 2:
            return False
        recent = history[-window:]
        if len(recent) < 2:
            return False
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return False
        return numerator / denominator > 0

    def is_above(self, threshold: float) -> bool:
        """Check if current temperature exceeds threshold."""
        temp = self.temperature
        if temp is None:
            return False
        return temp >= threshold

    def _poll_loop(self) -> None:
        """Background polling loop. Runs until stop() is called."""
        while not self._stop_event.is_set():
            temp = self._read_fn()
            with self._lock:
                self._current_temp = temp
                if temp is not None:
                    self._history.append(temp)
            self._stop_event.wait(self._poll_interval)
