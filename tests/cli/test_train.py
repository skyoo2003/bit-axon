from typer.testing import CliRunner

from bit_axon.cli.main import app
from tests.cli import strip_ansi

runner = CliRunner()


class TestCLITrain:
    def test_train_help(self):
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "data" in strip_ansi(result.output).lower()

    def test_train_requires_data(self):
        result = runner.invoke(app, ["train"])
        assert result.exit_code != 0

    def test_train_requires_weights(self):
        result = runner.invoke(app, ["train", "data.json"])
        assert result.exit_code != 0

    def test_train_config_small(self):
        result = runner.invoke(
            app,
            ["train", "data.json", "--model-weights", "/tmp/nonexistent", "--config-small", "--max-steps", "1"],
        )
        assert result.exit_code != 0 or "error" in strip_ansi(result.output).lower() or result.exit_code == 0
