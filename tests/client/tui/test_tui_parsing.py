import pytest
from sensai_tui.parsing import DEFAULT_URL, Parsing


class TestTuiParsing:
    def test_defaults_url(self) -> None:
        args = Parsing().parse_args(["--model", "qwen3:1.7b"])

        assert args.url == DEFAULT_URL == "ws://127.0.0.1:8000/ws"
        assert args.model == "qwen3:1.7b"

    def test_model_is_required(self) -> None:
        with pytest.raises(SystemExit):
            Parsing().parse_args([])

    def test_reads_url_and_model(self) -> None:
        args = Parsing().parse_args(["--url", "ws://host:9000/ws", "--model", "mistral"])

        assert args.url == "ws://host:9000/ws"
        assert args.model == "mistral"
