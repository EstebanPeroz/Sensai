import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import override
from unittest.mock import MagicMock, patch

import pytest

from sensai.config.settings import LoggingSettings
from sensai.core.events import EventBus
from sensai.history.conversation import Conversation
from sensai.llm.error import RequestCallError
from sensai.llm.persona import Persona
from sensai.llm.responses import ChatResponse
from sensai.llm.thinking.react import ANSWER_INSTRUCTION, ReAct
from sensai.memory.persistent_db.error import DatabaseError
from sensai.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai.monitoring.interaction import Interaction, InteractionStatus
from sensai.monitoring.logger import InteractionLogger
from sensai.pipeline import ChatPipeline
from sensai.pipeline.modules import ModelStage, PersonaStage
from sensai.tools.tool_manager import ToolManager
from tests.commands.fakes import FakeUI

_TIMEOUT = 5
_DONE = ChatResponse({"model": "m", "message": {}, "done": True, "prompt_eval_count": 11, "eval_count": 2})


def chunk(content: str) -> ChatResponse:
    return ChatResponse({"model": "m", "message": {"role": "assistant", "content": content}, "done": False})


def turn(thought: str, answer: str) -> list[Iterator[ChatResponse]]:
    """The two calls of a turn without tools: the model thinks, then answers."""
    return [iter([chunk(thought), _DONE]), iter([chunk(answer), _DONE])]


class InterruptingUI(FakeUI):
    """Reports a user event as soon as the conversation polls for one, i.e. the user hits a key mid-stream."""

    @override
    def wait_event(self, *, timeout: float | None = None) -> bool:
        return True


class Harness:
    """A conversation without tools on a real SQLite database, logged through a real bus, with a mocked provider.

    Its pipeline is the persona stage over the model, the two stages a plain exchange goes through.
    """

    def __init__(self, db: SqlAlchemy) -> None:
        self.db, self.bus, self.provider = db, EventBus(), MagicMock()
        self.provider.current_model.return_value = "m"
        self.persona_manager = MagicMock()
        self.persona_manager.active_persona.return_value = None
        provider_manager = MagicMock()
        provider_manager.provider.return_value = self.provider
        react = ReAct(provider_manager, ToolManager())
        pipeline = ChatPipeline(ModelStage(react)).use(PersonaStage(self.persona_manager))
        self.conversation = Conversation(provider_manager, self.persona_manager, pipeline, self.bus)
        self.conversation.add_conv_to_db(None, "test")

    def only(self) -> Interaction:
        interactions = self.db.interactions.get_all()
        assert len(interactions) == 1
        return interactions[0]


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    db = SqlAlchemy(f"sqlite:///{tmp_path / 'test.db'}")
    with patch("sensai.history.conversation.get_db", return_value=db):
        yield Harness(db)


@pytest.fixture
def logged(harness: Harness) -> Harness:
    InteractionLogger(harness.db.interactions, LoggingSettings()).attach(harness.bus)
    return harness


def test_logs_the_full_payload_response_and_metrics(logged: Harness) -> None:
    logged.persona_manager.active_persona.return_value = Persona(None, "analyst", "desc", "You are an analyst.")
    logged.provider.chat.side_effect = [*turn("Hello.", "A"), *turn("Second.", "B")]

    logged.conversation.chat("first", FakeUI())
    logged.conversation.chat("second", FakeUI())

    answer = logged.db.interactions.get_all()[-1]
    assert answer.request == logged.provider.chat.call_args.args[0]["messages"]
    assert answer.request == [
        {"role": "system", "content": "You are an analyst."},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "Second."},
        {"role": "user", "content": ANSWER_INSTRUCTION},
    ]
    assert (answer.status, answer.response, answer.persona) == (InteractionStatus.OK, "B", "analyst")
    assert answer.branch_id == logged.conversation.uuid
    assert (answer.metrics.prompt_tokens, answer.metrics.completion_tokens) == (11, 2)
    assert answer.first_token_ms is not None


def test_logs_every_call_of_a_turn(logged: Harness) -> None:
    logged.provider.chat.side_effect = turn("Hello.", "Hi!")

    logged.conversation.chat("hi", FakeUI())

    interactions = logged.db.interactions.get_all()
    assert [interaction.response for interaction in interactions] == ["Hello.", "Hi!"]
    assert all(interaction.status is InteractionStatus.OK for interaction in interactions)


def test_logs_a_connection_failure(logged: Harness) -> None:
    logged.provider.chat.side_effect = RequestCallError("connection refused")

    logged.conversation.chat("hi", FakeUI())

    assert logged.only().status is InteractionStatus.ERROR
    assert "connection refused" in (logged.only().error or "")


def _cut_stream() -> Iterator[ChatResponse]:
    yield chunk("par")
    msg = "stream cut"
    raise RequestCallError(msg)


@pytest.mark.parametrize(
    ("chunks", "error"),
    [
        (lambda: iter([chunk("par"), ChatResponse({"error": "model crashed"})]), "model crashed"),
        (_cut_stream, "stream cut"),
    ],
    ids=["error chunk", "failure mid-stream"],
)
def test_logs_a_stream_ending_in_error(
    logged: Harness, chunks: Callable[[], Iterator[ChatResponse]], error: str
) -> None:
    logged.provider.chat.return_value = chunks()

    logged.conversation.chat("hi", FakeUI())

    assert (logged.only().status, logged.only().response) == (InteractionStatus.ERROR, "par")
    assert error in (logged.only().error or "")


def test_logs_an_interrupted_stream(logged: Harness) -> None:
    resume, done = threading.Event(), threading.Event()
    logged.bus.subscribe(Interaction, lambda _: done.set())

    def chunks() -> Iterator[ChatResponse]:
        yield chunk("Once upon")
        resume.wait(_TIMEOUT)
        yield chunk(" a time")

    logged.provider.chat.return_value = chunks()
    logged.conversation.chat("tell me a story", InterruptingUI())
    resume.set()

    assert done.wait(_TIMEOUT)
    assert logged.only().status is InteractionStatus.INTERRUPTED


def test_a_storage_failure_does_not_break_the_chat(harness: Harness) -> None:
    failing_repository = MagicMock()
    failing_repository.add.side_effect = DatabaseError("disk full")
    InteractionLogger(failing_repository, LoggingSettings()).attach(harness.bus)
    harness.provider.chat.side_effect = turn("Greeting.", "Hello")
    ui = FakeUI()

    harness.conversation.chat("hi", ui)

    assert [m.content for m in harness.conversation.history.messages()] == ["hi", "Hello"]
    assert any("Failed to log interaction" in message for message in ui.messages)


def test_an_unexpected_listener_crash_does_not_hang_the_chat(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    crashes: list[type[BaseException]] = []
    crashed = threading.Event()
    monkeypatch.setattr(threading, "excepthook", lambda args: (crashes.append(args.exc_type), crashed.set()))

    def crash(_: Interaction) -> None:
        raise RuntimeError

    harness.bus.subscribe(Interaction, crash)
    harness.provider.chat.side_effect = turn("Greeting.", "Hello")
    chat = threading.Thread(target=harness.conversation.chat, args=("hi", FakeUI()), daemon=True)

    chat.start()
    chat.join(_TIMEOUT)

    assert not chat.is_alive()
    assert crashed.wait(_TIMEOUT)
    assert crashes == [RuntimeError]
