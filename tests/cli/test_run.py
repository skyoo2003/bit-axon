from typer.testing import CliRunner

from bit_axon.cli.main import app

runner = CliRunner()


class TestCLIRun:
    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "bit-axon" in result.output

    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Bit-Axon" in result.output

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "prompt" in result.output.lower()

    def test_run_config_small(self):
        result = runner.invoke(app, ["run", "Hello", "--config-small", "--max-tokens", "5"])
        assert result.exit_code == 0

    def test_run_chat_config_small(self):
        result = runner.invoke(app, ["run", "--config-small", "--chat"], input="exit\n")
        assert result.exit_code == 0

    def test_no_args_is_help(self):
        result = runner.invoke(app, [])
        assert result.exit_code in (0, 2)
        assert "help" in result.output.lower() or "Usage" in result.output
