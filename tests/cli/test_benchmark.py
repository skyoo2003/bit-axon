from typer.testing import CliRunner

from bit_axon.cli.main import app
from tests.cli import strip_ansi

runner = CliRunner()


class TestCLIBenchmark:
    def test_help(self):
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        clean = strip_ansi(result.output).lower()
        assert "seq-lengths" in clean or "seq_lengths" in clean

    def test_config_small(self):
        result = runner.invoke(app, ["benchmark", "--config-small", "--seq-lengths", "64,128", "--iterations", "1", "--warmup", "0"])
        assert result.exit_code == 0
        assert "tok/s" in strip_ansi(result.output).lower() or "Benchmark" in strip_ansi(result.output)
