from pathlib import Path

import pytest
from sensai_server.parsing import DEFAULT_PORT, HOST, Parsing


class TestParsing:
    def test_defaults(self) -> None:
        args = Parsing().parse_args(["--config", "settings.toml"])
        assert args.port == DEFAULT_PORT == 8000
        assert args.config == Path("settings.toml")

    def test_listens_on_loopback_only(self) -> None:
        assert HOST == "127.0.0.1"

    def test_config_is_required(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            Parsing().parse_args([])
        assert "--config" in capsys.readouterr().err

    def test_overrides(self) -> None:
        args = Parsing().parse_args(["--port", "9000", "--config", "x.toml"])
        assert args.port == 9000
        assert args.config == Path("x.toml")

    def test_host_is_rejected(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            Parsing().parse_args(["--host", "0.0.0.0", "--config", "x.toml"])  # noqa: S104
        assert "unrecognized arguments: --host" in capsys.readouterr().err
