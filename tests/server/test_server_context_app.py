from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sensai_protocol import (
    AiChunk,
    AskCompression,
    Chat,
    CompressionAnswer,
    CompressionChoice,
    ContextUsage,
    Login,
    ServerMessage,
    System,
    parse_server_message,
)
from sensai_server.app import app as app_module
from sensai_server.app import create_app
from sensai_server.app._state import app_state
from sensai_server.cache import SemanticCache
from sensai_server.chat.react import EFFORTS
from sensai_server.config.settings import ContextSettings, Settings
from sensai_server.monitoring.budget import ContextBudget
from sensai_server.monitoring.interaction import InteractionStatus
from sensai_server.pipeline.modules import CacheStage, CompressionStage, PersonaStage
from sensai_server.pipeline.modules.compression import SUMMARY_HEADER, SUMMARY_INSTRUCTION, Compressor
from sensai_server.pipeline.modules.guardrails import BadWordsGuard
from sensai_server.session.session import DEFAULT_EFFORT
from starlette.testclient import WebSocketTestSession

from tests.server.conftest import MODEL, settings_with_db
from tests.server.fakes import FakeAdapter, chunk

BUDGET = 100
THRESHOLD = 0.8


@pytest.fixture(autouse=True)
def single_call(monkeypatch: pytest.MonkeyPatch) -> None:
    # The default effort makes a single answering call per chat.
    monkeypatch.setitem(EFFORTS, DEFAULT_EFFORT, 0)


def _app(tmp_path: Path, context: ContextSettings | None = None) -> FastAPI:
    settings = settings_with_db(tmp_path)
    return create_app(Settings(llm=settings.llm, db=settings.db, context=context or ContextSettings()))


def _answer(content: str, prompt_tokens: int) -> list[dict]:
    """An Ollama stream answering `content`, its `done` chunk reporting `prompt_tokens`."""
    return [chunk(content), {**chunk("", done=True), "prompt_eval_count": prompt_tokens, "eval_count": 0}]


def _login(ws: WebSocketTestSession) -> list[ServerMessage]:
    """Log in, and return what the server sends: the session, then its usage."""
    ws.send_text(Login(model=MODEL).model_dump_json())
    return [parse_server_message(ws.receive_text()) for _ in range(2)]


def _chat(ws: WebSocketTestSession, prompt: str, answer: CompressionChoice = CompressionChoice.COMPRESS) -> list:
    """Send `prompt`, answering an `ask_compression` with `answer`, and return what the server sends until its usage."""
    ws.send_text(Chat(prompt=prompt).model_dump_json())
    received: list[ServerMessage] = []
    while not isinstance(message := parse_server_message(ws.receive_text()), ContextUsage) or not any(
        isinstance(item, AiChunk) and item.done for item in received
    ):
        received.append(message)
        if isinstance(message, AskCompression):
            ws.send_text(CompressionAnswer(request_id=message.request_id, answer=answer).model_dump_json())
    return [*received, message]


def _usages(messages: list) -> list[ContextUsage]:
    return [message for message in messages if isinstance(message, ContextUsage)]


class TestWiring:
    def test_app_state_has_the_budget_and_the_compressor(self, tmp_path: Path) -> None:
        state = app_state(_app(tmp_path))
        assert isinstance(state.budget, ContextBudget)
        assert isinstance(state.compressor, Compressor)

    def test_compression_runs_after_the_cache_and_before_the_persona(self, tmp_path: Path) -> None:
        state = app_state(_app(tmp_path))
        pipeline = app_module._build_pipeline(  # noqa: SLF001
            state.react,
            None,
            cast("SemanticCache", object()),
            [],
            compressor=state.compressor,
            budget=state.budget,
        )
        stages: list[type] = [type(stage) for stage in pipeline._stages]  # noqa: SLF001
        assert stages == [BadWordsGuard, CacheStage, CompressionStage, PersonaStage]

    def test_compress_command_is_registered(self, tmp_path: Path) -> None:
        assert "/compress" in app_state(_app(tmp_path)).commands.names()

    def test_the_budget_asks_the_model_context_window(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(FakeAdapter, "replies", _answer("Hi", 10))
        app = _app(tmp_path, ContextSettings(budget=BUDGET))
        with TestClient(app) as test_client, test_client.websocket_connect("/ws") as ws:
            _login(ws)
            _chat(ws, "hello")
        assert FakeAdapter.last_payload is not None
        assert FakeAdapter.last_payload["options"] == {"num_ctx": BUDGET}

    def test_without_budget_the_gauge_is_the_model_context_length(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(FakeAdapter, "context_length", 4096)
        with TestClient(_app(tmp_path)) as test_client, test_client.websocket_connect("/ws") as ws:
            assert _usages(_login(ws)) == [ContextUsage(used=0, budget=4096, warning_threshold=THRESHOLD)]


class TestThroughWs:
    def test_the_gauge_rises_then_the_branch_is_compressed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            FakeAdapter,
            "script",
            [_answer("One.", 50), _answer("Two.", 90), _answer("A short summary.", 70), _answer("Three.", 40)],
        )
        app = _app(tmp_path, ContextSettings(budget=BUDGET, warning_threshold=THRESHOLD, keep_recent=2))

        with TestClient(app) as test_client, test_client.websocket_connect("/ws") as ws:
            assert _usages(_login(ws)) == [ContextUsage(used=0, budget=BUDGET, warning_threshold=THRESHOLD)]
            first = _chat(ws, "one")
            second = _chat(ws, "two")
            third = _chat(ws, "three")

        assert [usage.used for usage in _usages(first) + _usages(second)] == [50, 90]
        assert not any(isinstance(message, AskCompression) for message in first + second)
        [asked] = [message for message in third if isinstance(message, AskCompression)]
        assert asked.used == 90
        assert System(message="Summarised the 2 oldest messages of the conversation: 70 -> 0 tokens.") in third
        assert _usages(third)[-1].used == 40

        summary_request, answer_request = FakeAdapter.payloads[2:]
        assert SUMMARY_INSTRUCTION in summary_request["messages"][0]["content"]
        assert summary_request["options"] == {"num_ctx": BUDGET}
        sent = [message["content"] for message in answer_request["messages"]]
        assert f"{SUMMARY_HEADER}\nA short summary." in sent
        assert "one" not in sent

        state = app_state(app)
        interactions = state.db.interactions.get_all()
        [summary_call] = [interaction for interaction in interactions if interaction.branch_id is None]
        assert summary_call.status is InteractionStatus.OK
        assert summary_call.response == "A short summary."
        [branch_id] = {interaction.branch_id for interaction in interactions if interaction.branch_id is not None}
        summary = state.db.summaries.get(branch_id)
        assert summary is not None
        assert summary.covered == 2
        assert summary.content == "A short summary."

    def test_a_new_conversation_saves_the_prompt_in_a_new_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(FakeAdapter, "script", [_answer("One.", 90), _answer("Two.", 10)])
        app = _app(tmp_path, ContextSettings(budget=BUDGET, warning_threshold=THRESHOLD, keep_recent=2))

        with TestClient(app) as test_client, test_client.websocket_connect("/ws") as ws:
            _login(ws)
            _chat(ws, "one")
            second = _chat(ws, "two", CompressionChoice.NEW_CONVERSATION)

        assert any(isinstance(message, AskCompression) for message in second)
        assert _usages(second)[-1].used == 10
        assert len(FakeAdapter.payloads) == 2
        sent = [message["content"] for message in FakeAdapter.payloads[1]["messages"]]
        assert "two" in sent
        assert "one" not in sent
        state = app_state(app)
        branches = {interaction.branch_id for interaction in state.db.interactions.get_all()}
        assert len(branches) == 2
        assert None not in branches
