from typer.testing import CliRunner

from bit_axon.cli.main import app

runner = CliRunner()


class TestCLIPipeline:
    def test_help(self):
        result = runner.invoke(app, ["pipeline", "--help"])
        assert result.exit_code == 0
        assert "pipeline" in result.output.lower()
        assert "train" in result.output.lower()

    def test_short_run(self, tmp_path):
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
