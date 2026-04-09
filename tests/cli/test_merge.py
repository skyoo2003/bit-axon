from typer.testing import CliRunner

from bit_axon.cli.main import app
from tests.cli import strip_ansi

runner = CliRunner()


class TestCLIMerge:
    def test_help(self):
        result = runner.invoke(app, ["merge", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output).lower()
        assert "base-model" in output or "base_model" in output
