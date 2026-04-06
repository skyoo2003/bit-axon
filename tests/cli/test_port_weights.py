from typer.testing import CliRunner

from bit_axon.cli.main import app

runner = CliRunner()


class TestCLIPortWeights:
    def test_help(self):
        result = runner.invoke(app, ["port-weights", "--help"])
        assert result.exit_code == 0
        assert "Qwen" in result.output
        assert "output" in result.output.lower()

    def test_config_small(self, tmp_path):
        result = runner.invoke(app, ["port-weights", str(tmp_path / "ported"), "--config-small"])
        assert result.exit_code == 0
        assert "saved" in result.output.lower()
