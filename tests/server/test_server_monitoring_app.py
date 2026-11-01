import asyncio
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sensai_protocol import AiChunk, Chat, Login, ServerMessage, parse_server_message
from sensai_server.app import _handlers as handlers
from sensai_server.app import create_app
from sensai_server.app._state import app_state
from sensai_server.chat.react import EFFORTS
from sensai_server.config.settings import LoggingSettings, Settings
from sensai_server.monitoring.interaction import Interaction, InteractionStatus
from sensai_server.monitoring.logger import InteractionLogger
from sensai_server.session.session import DEFAULT_EFFORT
from starlette.testclient import WebSocketTestSession

from tests.server.conftest import MODEL, settings_with_db
from tests.server.fakes import FakeAdapter, RecordingOutbound, chunk

TIMEOUT = 5


class StreamingOutbound(RecordingOutbound):
    """A `RecordingOutbound` setting `streaming` once the first chunk of an answer is sent."""

    def __init__(self) -> None:
        super().__init__()
        self.streaming = asyncio.Event()

    async def send(self, message: ServerMessage) -> bool:
        if isinstance(message, AiChunk):
            self.streaming.set()
        return await super().send(message)


@pytest.fixture(autouse=True)
def replies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FakeAdapter, "replies", [chunk("Hel"), chunk("lo", done=True)])
    # The default effort makes a single answering call per chat.
    monkeypatch.setitem(EFFORTS, DEFAULT_EFFORT, 0)


def _app(tmp_path: Path, logging: LoggingSettings | None = None) -> FastAPI:
    settings = settings_with_db(tmp_path)
    return create_app(Settings(llm=settings.llm, db=settings.db, logging=logging or LoggingSettings()))


def _stored(app: FastAPI) -> list[Interaction]:
    return app_state(app).db.interactions.get_all()


def _chat(ws: WebSocketTestSession, prompt: str) -> None:
    """Log in, send `prompt` and read until the last chunk of its answer."""
    ws.send_text(Login(model=MODEL).model_dump_json())
    ws.receive_text()
    ws.send_text(Chat(prompt=prompt).model_dump_json())
    while not (isinstance(message := parse_server_message(ws.receive_text()), AiChunk) and message.done):
        pass


class TestWiring:
    def test_app_state_has_the_logger(self, tmp_path: Path) -> None:
        assert isinstance(app_state(_app(tmp_path)).interaction_logger, InteractionLogger)


class TestStorage:
    def test_chat_through_ws_is_stored(self, tmp_path: Path) -> None:
        app = _app(tmp_path)
        with TestClient(app) as test_client, test_client.websocket_connect("/ws") as ws:
            _chat(ws, "hello")
        [interaction] = _stored(app)
        assert interaction.status is InteractionStatus.OK
        assert interaction.model == MODEL
        assert interaction.response == "Hello"
        assert interaction.branch_id is not None

    def test_disabled_logging_stores_nothing(self, tmp_path: Path) -> None:
        app = _app(tmp_path, LoggingSettings(enabled=False))
        with TestClient(app) as test_client, test_client.websocket_connect("/ws") as ws:
            _chat(ws, "hello")
        assert _stored(app) == []

    def test_log_content_off_keeps_only_metadata(self, tmp_path: Path) -> None:
        app = _app(tmp_path, LoggingSettings(log_content=False))
        with TestClient(app) as test_client, test_client.websocket_connect("/ws") as ws:
            _chat(ws, "hello")
        [interaction] = _stored(app)
        assert interaction.status is InteractionStatus.OK
        assert interaction.request is None
        assert interaction.response is None


class TestShutdown:
    async def test_stopping_mid_answer_stores_the_interrupted_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        block = threading.Event()
        monkeypatch.setattr(FakeAdapter, "block", block)
        app = _app(tmp_path)
        state = app_state(app)
        out = StreamingOutbound()
        try:
            async with app.router.lifespan_context(app):
                session = await state.registry.open(out, MODEL)
                assert session is not None
                await handlers.handle_chat(session, Chat(prompt="hello"), state.pipeline, state.db, state.events)
                await asyncio.wait_for(out.streaming.wait(), TIMEOUT)
            # The lifespan closed the session, then waited for the write: the row is already there.
            assert len(state.registry) == 0
            [interaction] = _stored(app)
        finally:
            block.set()
        assert interaction.status is InteractionStatus.INTERRUPTED
        assert interaction.response == "Hel"
