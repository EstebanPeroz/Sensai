import pytest
from sensai_tui import _run
from sensai_tui.ui import ChatApp

from tests.client.tui.fakes import FakeConnection

MODEL = "qwen3:1.7b"


class TestRun:
    async def test_cannot_connect_prints_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        async def fail(_self: ChatApp) -> None:
            pytest.fail("the app must not start without a connection")

        monkeypatch.setattr(ChatApp, "run_async", fail)
        assert await _run(FakeConnection(fail_open=True), MODEL) == 1
        assert "cannot connect to ws://fake/ws: connection refused" in capsys.readouterr().err

    async def test_connected_runs_app(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = FakeConnection()
        started: list[ChatApp] = []

        async def run(self: ChatApp) -> None:
            assert conn.opened
            started.append(self)

        monkeypatch.setattr(ChatApp, "run_async", run)
        assert await _run(conn, MODEL) == 0
        assert len(started) == 1
