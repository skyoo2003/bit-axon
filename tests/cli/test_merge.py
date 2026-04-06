from typer.testing import CliRunner

from bit_axon.cli.main import app

runner = CliRunner()


class TestCLIMerge:
    def test_help(self):
        result = runner.invoke(app, ["merge", "--help"])
        assert result.exit_code == 0
        assert "base-model" in result.output.lower() or "base_model" in result.output.lower()
