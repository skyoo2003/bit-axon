from typer.testing import CliRunner

from bit_axon.cli.main import app

runner = CliRunner()


class TestCLIBenchmark:
    def test_help(self):
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "seq-lengths" in result.output.lower() or "seq_lengths" in result.output.lower()

    def test_config_small(self):
        result = runner.invoke(app, ["benchmark", "--config-small", "--seq-lengths", "64,128", "--iterations", "1", "--warmup", "0"])
        assert result.exit_code == 0
        assert "tok/s" in result.output.lower() or "Benchmark" in result.output
