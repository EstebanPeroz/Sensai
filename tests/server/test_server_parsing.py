from pathlib import Path

from sensai_server.parsing import DEFAULT_HOST, DEFAULT_MODELS, DEFAULT_PORT, Parsing


class TestParsing:
    def test_defaults(self) -> None:
        args = Parsing().parse_args([])
        assert args.host == DEFAULT_HOST == "127.0.0.1"
        assert args.port == DEFAULT_PORT == 8000
        assert args.config is None

    def test_overrides(self) -> None:
        args = Parsing().parse_args(["--host", "0.0.0.0", "--port", "9000", "--config", "x.toml"])  # noqa: S104
        assert args.host == "0.0.0.0"  # noqa: S104
        assert args.port == 9000
        assert args.config == Path("x.toml")

    def test_models(self) -> None:
        assert Parsing().parse_args([]).models == DEFAULT_MODELS == ["qwen3:1.7b"]
        assert Parsing().parse_args(["--model", "a", "--model", "b"]).models == ["a", "b"]
