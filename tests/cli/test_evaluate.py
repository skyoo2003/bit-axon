from typer.testing import CliRunner

from bit_axon.cli.main import app
from tests.cli import strip_ansi

runner = CliRunner()


class TestCLIEvaluate:
    def test_help(self):
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "perplexity" in output.lower()
        assert "WikiText" in output

    def test_config_small(self):
        result = runner.invoke(app, ["evaluate", "/tmp/nonexistent", "--config-small"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "Perplexity" in output
        assert "PPL" in output or "perplexity" in output.lower()
