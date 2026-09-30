from pathlib import Path

from sensai.parsing import parse_args


class TestParseArgs:
    def test_defaults_to_no_override(self) -> None:
        args = parse_args([])

        assert args.config is None
        assert args.model is None

    def test_reads_config_and_model(self) -> None:
        args = parse_args(["--config", "custom.toml", "--model", "mistral"])

        assert args.config == Path("custom.toml")
        assert args.model == "mistral"
