from unittest.mock import patch

from typer.testing import CliRunner

from bit_axon.cli.main import app
from tests.cli import strip_ansi

runner = CliRunner()


class TestCLIDownload:
    def test_help(self):
        result = runner.invoke(app, ["download", "--help"])
        assert result.exit_code == 0
        assert "repo" in strip_ansi(result.output).lower()

    def test_download_calls_snapshot(self):
        with patch("huggingface_hub.snapshot_download", return_value="/fake/path") as mock_dl:
            result = runner.invoke(app, ["download", "some/repo", "--local-dir", "/tmp/test"])
            assert result.exit_code == 0
            mock_dl.assert_called_once_with(repo_id="some/repo", local_dir="/tmp/test")
