"""End-to-end CLI integration tests."""

from typer.testing import CliRunner

from bit_axon.cli.main import app
from tests.cli import strip_ansi

runner = CliRunner()


class TestCLIIntegration:
    def test_all_commands_help(self):
        for cmd in ["run", "train", "quantize", "merge", "benchmark", "download"]:
            result = runner.invoke(app, [cmd, "--help"])
            assert result.exit_code == 0, f"{cmd} --help failed: {strip_ansi(result.output)}"

    def test_version_consistency(self):
        from bit_axon import __version__

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in strip_ansi(result.output)

    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        assert result.exit_code in (0, 2)

    def test_error_messages_user_friendly(self):
        result = runner.invoke(app, ["run", "--config-small"])
        output = strip_ansi(result.output).lower()
        assert "Traceback" not in strip_ansi(result.output)
        assert "error" in output or "usage" in output or "help" in output

    def test_unknown_command(self):
        result = runner.invoke(app, ["nonexistent"])
        assert result.exit_code != 0

    def test_benchmark_config_small_e2e(self):
        result = runner.invoke(app, ["benchmark", "--config-small", "--seq-lengths", "64", "--iterations", "1", "--warmup", "0"])
        assert result.exit_code == 0

    def test_run_config_small_e2e(self):
        result = runner.invoke(app, ["run", "test prompt", "--config-small", "--max-tokens", "3"])
        assert result.exit_code == 0

    def test_download_help_shows_default_repo(self):
        result = runner.invoke(app, ["download", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "skyoo2003" in output or "huggingface" in output.lower()
