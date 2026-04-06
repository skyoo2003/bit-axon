from typer.testing import CliRunner

from bit_axon.cli.main import app

runner = CliRunner()


class TestCLIEvaluate:
    def test_help(self):
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "perplexity" in result.output.lower()
        assert "WikiText" in result.output

    def test_config_small(self):
        result = runner.invoke(app, ["evaluate", "/tmp/nonexistent", "--config-small"])
        assert result.exit_code == 0
        assert "Perplexity" in result.output
        assert "PPL" in result.output or "perplexity" in result.output.lower()
