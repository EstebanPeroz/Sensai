from pathlib import Path
from typing import Any

import pytest
import sensai_server
from fastapi import FastAPI
from sensai_server import main
from sensai_server.app._state import app_state
from sensai_server.config.settings import OllamaSettings, Settings


class FakeUvicorn:
    """Records the app and address `run` was called with, instead of serving."""

    def __init__(self) -> None:
        self.calls: list[tuple[FastAPI, dict[str, Any]]] = []

    def run(self, app: FastAPI, **kwargs: Any) -> None:  # noqa: ANN401 - mirrors uvicorn.run
        self.calls.append((app, kwargs))


@pytest.fixture
def uvicorn(monkeypatch: pytest.MonkeyPatch) -> FakeUvicorn:
    fake = FakeUvicorn()
    monkeypatch.setattr(sensai_server, "uvicorn", fake)
    return fake


class TestMain:
    def test_serves_an_app_holding_the_loaded_settings(self, tmp_path: Path, uvicorn: FakeUvicorn) -> None:
        path = tmp_path / "settings.toml"
        db = tmp_path / "test.db"
        path.write_text(f'[llm.ollama]\nbase_url = "http://ollama:1234"\n[db.persistent]\nurl = "sqlite:///{db}"\n')

        assert main(["--config", str(path), "--host", "0.0.0.0", "--port", "9000"]) == 0  # noqa: S104

        [(app, address)] = uvicorn.calls
        assert address == {"host": "0.0.0.0", "port": 9000}  # noqa: S104
        settings: Settings = app_state(app).settings
        assert settings.llm.providers == {"ollama": OllamaSettings(base_url="http://ollama:1234")}

    def test_missing_config_fails_without_serving(
        self, tmp_path: Path, uvicorn: FakeUvicorn, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--config", str(tmp_path / "nope.toml")]) == 1

        assert uvicorn.calls == []
        assert "config file not found" in capsys.readouterr().err

    def test_invalid_config_fails_without_serving(
        self, tmp_path: Path, uvicorn: FakeUvicorn, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "settings.toml"
        path.write_text("[cache]\nttl = 0\n")

        assert main(["--config", str(path)]) == 1

        assert uvicorn.calls == []
        assert "invalid settings" in capsys.readouterr().err
