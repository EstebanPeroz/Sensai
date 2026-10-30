import pytest
from sensai_tui.parsing import DEFAULT_PORT, Parsing, server_url


class TestTuiParsing:
    def test_defaults_port(self) -> None:
        args = Parsing().parse_args(["--model", "qwen3:1.7b"])

        assert args.port == DEFAULT_PORT == 8000
        assert args.model == "qwen3:1.7b"

    def test_model_is_required(self) -> None:
        with pytest.raises(SystemExit):
            Parsing().parse_args([])

    def test_reads_port_and_model(self) -> None:
        args = Parsing().parse_args(["--port", "9000", "--model", "mistral"])

        assert args.port == 9000
        assert args.model == "mistral"

    def test_url_is_rejected(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            Parsing().parse_args(["--url", "ws://host:9000/ws", "--model", "mistral"])
        assert "unrecognized arguments: --url" in capsys.readouterr().err


class TestServerUrl:
    def test_points_at_the_local_server(self) -> None:
        assert server_url(DEFAULT_PORT) == "ws://127.0.0.1:8000/ws"
        assert server_url(9000) == "ws://127.0.0.1:9000/ws"
