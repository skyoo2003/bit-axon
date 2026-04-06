"""End-to-end CLI integration tests."""

from typer.testing import CliRunner

from bit_axon.cli.main import app

runner = CliRunner()


class TestCLIIntegration:
    def test_all_commands_help(self):
        for cmd in ["run", "train", "quantize", "merge", "benchmark", "download"]:
            result = runner.invoke(app, [cmd, "--help"])
            assert result.exit_code == 0, f"{cmd} --help failed: {result.output}"

    def test_version_consistency(self):
        from bit_axon import __version__

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        assert result.exit_code in (0, 2)

    def test_error_messages_user_friendly(self):
        result = runner.invoke(app, ["run", "--config-small"])
        assert "Traceback" not in result.output
        assert "error" in result.output.lower() or "usage" in result.output.lower() or "help" in result.output.lower()

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
        assert "skyoo2003" in result.output or "huggingface" in result.output.lower()
