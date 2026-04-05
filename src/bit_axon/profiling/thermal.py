"""SoC thermal monitoring for Apple Silicon."""

import re
import subprocess


class ThermalMonitor:
    """Read Apple Silicon SoC temperature via powermetrics."""

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
