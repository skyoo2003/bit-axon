from unittest.mock import MagicMock

from typer.testing import CliRunner

from bit_axon.cli.main import app

runner = CliRunner()


def _make_fake_dataset(rows):
    ds = MagicMock()
    ds.__len__ = lambda self: len(rows)
    ds.__getitem__ = lambda self, idx: rows[idx]

    def _select(indices):
        subset = [rows[i] for i in indices]
        return _make_fake_dataset(subset)

    ds.select = _select
    return ds


class TestCLIPipeline:
    def test_help(self):
        result = runner.invoke(app, ["pipeline", "--help"], color=False)
        assert result.exit_code == 0
        assert "pipeline" in result.output.lower()
        assert "train" in result.output.lower()
        assert "--sft-data" in result.output
        assert "--orpo-data" in result.output
        assert "--tokenizer" in result.output

    def test_short_run(self, tmp_path):
        """Backward compat: pipeline with no dataset args uses SimpleDataset fallback."""
        result = runner.invoke(
            app,
            [
                "pipeline",
                "--output-dir",
                str(tmp_path / "pipe"),
                "--max-steps",
                "5",
                "--orpo-steps",
                "3",
                "--max-seq-len",
                "16",
            ],
        )
        assert result.exit_code == 0
        assert "Pipeline Results" in result.output or "complete" in result.output.lower()

    def test_sft_data_requires_tokenizer(self):
        """--sft-data without --tokenizer should exit with error."""
        result = runner.invoke(
            app,
            [
                "pipeline",
                "--output-dir",
                "/tmp/test_pipe",
                "--max-steps",
                "1",
                "--orpo-steps",
                "1",
                "--sft-data",
                "alpaca",
            ],
        )
        assert result.exit_code != 0
        assert "tokenizer" in result.output.lower()

    def test_orpo_data_requires_tokenizer(self):
        """--orpo-data without --tokenizer should exit with error."""
        result = runner.invoke(
            app,
            [
                "pipeline",
                "--output-dir",
                "/tmp/test_pipe",
                "--max-steps",
                "1",
                "--orpo-steps",
                "1",
                "--orpo-data",
                "ultrafeedback",
            ],
        )
        assert result.exit_code != 0
        assert "tokenizer" in result.output.lower()
