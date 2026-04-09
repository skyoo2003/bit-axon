from typer.testing import CliRunner

from bit_axon.cli.main import app
from tests.cli import strip_ansi

runner = CliRunner()


class TestCLIQuantize:
    def test_help(self):
        result = runner.invoke(app, ["quantize", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output).lower()
        assert "model-path" in output or "model_path" in output

    def test_config_small(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        result = runner.invoke(app, ["quantize", str(model_dir), "--config-small", "-o", str(tmp_path / "output")])
        assert result.exit_code == 0
        assert (tmp_path / "output" / "weights.safetensors").exists()
