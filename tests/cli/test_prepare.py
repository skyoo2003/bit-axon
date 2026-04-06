import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from bit_axon.cli.main import app

runner = CliRunner()

FAKE_ROWS = [
    {"instruction": "Translate to Korean", "input": "Hello", "output": "안녕하세요"},
    {"instruction": "Summarize", "input": "", "output": "A brief summary"},
]


def _make_fake_dataset(rows):
    ds = MagicMock()
    ds.__len__ = lambda self: len(rows)
    ds.__getitem__ = lambda self, idx: rows[idx]

    def _select(indices):
        subset = [rows[i] for i in indices]
        return _make_fake_dataset(subset)

    ds.select = _select
    return ds


class TestCLIPrepare:
    def test_help(self):
        result = runner.invoke(app, ["prepare", "--help"])
        assert result.exit_code == 0
        assert "HuggingFace" in result.output

    def test_prepare_alpaca_format(self, tmp_path):
        fake_ds = _make_fake_dataset(FAKE_ROWS)
        out = str(tmp_path / "out.jsonl")
        with patch("datasets.load_dataset", return_value=fake_ds):
            result = runner.invoke(app, ["prepare", "some/repo", "--output", out, "--format", "alpaca"])
        assert result.exit_code == 0
        lines = (tmp_path / "out.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert "instruction" in parsed

    def test_prepare_messages_format(self, tmp_path):
        fake_ds = _make_fake_dataset(FAKE_ROWS)
        out = str(tmp_path / "out.jsonl")
        with patch("datasets.load_dataset", return_value=fake_ds):
            result = runner.invoke(app, ["prepare", "some/repo", "--output", out, "--format", "messages"])
        assert result.exit_code == 0
        parsed = json.loads((tmp_path / "out.jsonl").read_text().strip().split("\n")[0])
        assert "messages" in parsed
        assert parsed["messages"][0]["role"] == "user"
        assert parsed["messages"][1]["content"] == "안녕하세요"

    def test_prepare_with_limit(self, tmp_path):
        fake_ds = _make_fake_dataset(FAKE_ROWS)
        out = str(tmp_path / "out.jsonl")
        with patch("datasets.load_dataset", return_value=fake_ds):
            result = runner.invoke(app, ["prepare", "some/repo", "--output", out, "--limit", "1"])
        assert result.exit_code == 0
        lines = (tmp_path / "out.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1

    def test_prepare_auto_output_name(self, tmp_path):
        fake_ds = _make_fake_dataset(FAKE_ROWS)
        with patch("datasets.load_dataset", return_value=fake_ds), patch("bit_axon.cli.prepare.Path.mkdir"):
            result = runner.invoke(app, ["prepare", "org/repo"])
        assert result.exit_code == 0
        assert "org_repo_train.jsonl" in result.output

    def test_prepare_invalid_format(self):
        result = runner.invoke(app, ["prepare", "some/repo", "--format", "invalid"])
        assert result.exit_code != 0

    def test_prepare_creates_output_directory(self, tmp_path):
        fake_ds = _make_fake_dataset(FAKE_ROWS)
        out = str(tmp_path / "subdir" / "deep" / "out.jsonl")
        with patch("datasets.load_dataset", return_value=fake_ds):
            result = runner.invoke(app, ["prepare", "some/repo", "--output", out])
        assert result.exit_code == 0
        assert (tmp_path / "subdir" / "deep" / "out.jsonl").exists()
