from typer.testing import CliRunner

from bit_axon.cli.main import app
from tests.cli import strip_ansi

runner = CliRunner()


class TestCLIPortWeights:
    def test_help(self):
        result = runner.invoke(app, ["port-weights", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "Qwen" in output
        assert "output" in output.lower()

    def test_config_small(self, tmp_path):
        result = runner.invoke(app, ["port-weights", str(tmp_path / "ported"), "--config-small"])
        assert result.exit_code == 0
        assert "saved" in strip_ansi(result.output).lower()
