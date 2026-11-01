import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest
from sensai_protocol import (
    AskCompression,
    CompressionChoice,
    ContextUsage,
    History,
    ServerMessage,
    System,
)
from sensai_protocol.server_messages import Role
from sensai_server.cache import SessionCache
from sensai_server.chat import chat
from sensai_server.chat.react import ReAct
from sensai_server.commands import CommandRegistry, build_commands
from sensai_server.commands._compress import Compress
from sensai_server.config.settings import ContextSettings
from sensai_server.events import EventBus
from sensai_server.llm.persona import Persona, PersonaManager
from sensai_server.llm.provider import ProviderManager
from sensai_server.llm.responses import ChatMetrics, ChatResponse
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai_server.memory.summary import Summary
from sensai_server.monitoring.budget import ContextBudget
from sensai_server.monitoring.interaction import Interaction, InteractionStatus
from sensai_server.pipeline import ChatPipeline
from sensai_server.pipeline.modules import CompressionStage, ModelStage
from sensai_server.pipeline.modules.compression import Compression, CompressionError, Compressor
from sensai_server.session import Session, SessionRegistry
from sensai_server.tools.tool_manager import ToolManager

from tests.server.conftest import MODEL, SETTINGS, db_url
from tests.server.fakes import FakeAdapter, FakeSessionCache, RecordingOutbound, chunk

BUDGET = 100
NEAR = 90
FAR = 10


class AnsweringOutbound(RecordingOutbound):
    """A `RecordingOutbound` whose client answers every `ask_compression` with `answer`.

    With `answer` None, the question cannot be sent, as on a closed socket.
    """

    def __init__(self, answer: CompressionChoice | None = None) -> None:
        super().__init__()
        self.answer = answer

    async def send(self, message: ServerMessage) -> bool:
        if isinstance(message, AskCompression) and self.answer is None:
            return False
        await super().send(message)
        if isinstance(message, AskCompression):
            self.resolve(message.request_id, self.answer)
        return True


class FakeCompressor:
    """A `Compressor` returning `result`, or raising it, and recording what it was asked."""

    def __init__(self, result: Compression | CompressionError | None) -> None:
        self._result = result
        self.calls: list[tuple[UUID, list[dict], str, str | None]] = []

    async def compress(
        self, branch_id: UUID, messages: list[dict], model: str, persona: str | None = None
    ) -> Compression | None:
        self.calls.append((branch_id, messages, model, persona))
        if isinstance(self._result, CompressionError):
            raise self._result
        return self._result


@pytest.fixture
def db(tmp_path: Path) -> SqlAlchemy:
    return SqlAlchemy(db_url(tmp_path))


@pytest.fixture
def pm() -> ProviderManager:
    return ProviderManager(SETTINGS.llm)


@pytest.fixture
def events() -> EventBus:
    return EventBus()


@pytest.fixture
def interactions(events: EventBus) -> list[Interaction]:
    recorded: list[Interaction] = []
    events.subscribe(Interaction, recorded.append)
    return recorded


@pytest.fixture
def budget(events: EventBus) -> ContextBudget:
    budget = ContextBudget(ContextSettings(budget=BUDGET), lambda _: None)
    budget.attach(events)
    return budget


@pytest.fixture
def compressor(pm: ProviderManager, db: SqlAlchemy, events: EventBus) -> Compressor:
    return Compressor(pm, db.summaries, keep_recent=2, on_interaction=events.emit)


@pytest.fixture
def replies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FakeAdapter, "replies", [chunk("Hi"), chunk("", done=True)])


async def _open(db: SqlAlchemy, pm: ProviderManager, out: RecordingOutbound) -> Session:
    registry = SessionRegistry(pm, PersonaManager(db), ToolManager())
    opened = await registry.open(out, MODEL)
    assert opened is not None
    opened.effort = "off"
    out.sent.clear()
    return opened


def _sent(session: Session) -> list[ServerMessage]:
    return cast("RecordingOutbound", session.out).sent


def _systems(session: Session) -> list[str]:
    return [message.message for message in _sent(session) if isinstance(message, System)]


def _measure(events: EventBus, branch_id: UUID, tokens: int) -> None:
    """Emit a chat call of `tokens` prompt tokens on `branch_id`, as `run_chat` does."""
    events.emit(
        Interaction(
            model=MODEL,
            status=InteractionStatus.OK,
            latency_ms=1.0,
            branch_id=branch_id,
            metrics=ChatMetrics(prompt_tokens=tokens, completion_tokens=0),
        )
    )


async def _started(session: Session, db: SqlAlchemy, exchanges: int = 3) -> UUID:
    """Give the session a branch of `exchanges` user and assistant messages, and return its id."""
    session.conv.add_conv_to_db(db)
    assert session.conv.uuid is not None
    for n in range(exchanges):
        await session.conv.save_message(Role.USER, f"question {n}", session.out, db)
        await session.conv.save_message(Role.ASSISTANT, f"answer {n}", session.out, db)
    _sent(session).clear()
    return session.conv.uuid


class TestCompressCommand:
    async def test_summarises_the_current_branch(self, db: SqlAlchemy, pm: ProviderManager) -> None:
        session = await _open(db, pm, RecordingOutbound())
        branch = await _started(session, db, exchanges=1)
        fake = FakeCompressor(Compression(4, 900, 40))

        await Compress(cast("Compressor", fake)).execute(session)

        assert fake.calls == [
            (
                branch,
                [{"role": "user", "content": "question 0"}, {"role": "assistant", "content": "answer 0"}],
                MODEL,
                None,
            )
        ]
        assert _systems(session) == ["Summarised the 4 oldest messages of the conversation: 900 -> 40 tokens."]

    async def test_passes_the_persona(self, db: SqlAlchemy, pm: ProviderManager) -> None:
        session = await _open(db, pm, RecordingOutbound())
        await _started(session, db)
        session.conv.persona = Persona(uuid=None, name="tutor", description="A tutor.", prompt="Teach.")
        fake = FakeCompressor(None)

        await Compress(cast("Compressor", fake)).execute(session)

        assert fake.calls[0][3] == "tutor"

    async def test_nothing_left_to_summarise(self, db: SqlAlchemy, pm: ProviderManager) -> None:
        session = await _open(db, pm, RecordingOutbound())
        await _started(session, db)

        await Compress(cast("Compressor", FakeCompressor(None))).execute(session)

        assert _systems(session) == ["Nothing to compress: only the last messages are left to summarise."]

    async def test_a_conversation_not_started(self, db: SqlAlchemy, pm: ProviderManager) -> None:
        session = await _open(db, pm, RecordingOutbound())
        fake = FakeCompressor(None)

        await Compress(cast("Compressor", fake)).execute(session)

        assert fake.calls == []
        assert _systems(session) == ["Nothing to compress: the conversation has not started."]

    async def test_reports_a_failure(self, db: SqlAlchemy, pm: ProviderManager) -> None:
        session = await _open(db, pm, RecordingOutbound())
        await _started(session, db)
        error = CompressionError("model not found")

        await Compress(cast("Compressor", FakeCompressor(error))).execute(session)

        assert _systems(session) == [f"The conversation could not be compressed: {error.message}"]

    async def test_takes_no_argument(self, db: SqlAlchemy, pm: ProviderManager) -> None:
        session = await _open(db, pm, RecordingOutbound())
        await _started(session, db)
        fake = FakeCompressor(None)

        await Compress(cast("Compressor", fake)).execute(session, "now")

        assert fake.calls == []
        assert _systems(session) == ["This command takes no argument. Usage: /compress"]

    async def test_then_sends_the_estimated_usage(
        self, db: SqlAlchemy, pm: ProviderManager, budget: ContextBudget, events: EventBus
    ) -> None:
        session = await _open(db, pm, RecordingOutbound())
        branch = await _started(session, db)
        _measure(events, branch, NEAR)

        await Compress(cast("Compressor", FakeCompressor(Compression(4, 60, 10))), budget).execute(session)

        assert isinstance(_sent(session)[0], System)
        assert _sent(session)[1:] == [ContextUsage(used=NEAR - 60 + 10, budget=BUDGET, warning_threshold=0.8)]

    @pytest.mark.parametrize("result", [None, Compression(4, None, None), CompressionError("down")])
    async def test_sends_the_usage_as_measured_without_token_counts(
        self,
        db: SqlAlchemy,
        pm: ProviderManager,
        budget: ContextBudget,
        events: EventBus,
        result: Compression | CompressionError | None,
    ) -> None:
        session = await _open(db, pm, RecordingOutbound())
        branch = await _started(session, db)
        _measure(events, branch, NEAR)

        await Compress(cast("Compressor", FakeCompressor(result)), budget).execute(session)

        assert _sent(session)[-1] == ContextUsage(used=NEAR, budget=BUDGET, warning_threshold=0.8)

    @pytest.mark.usefixtures("replies")
    async def test_stores_the_summary_and_records_the_call(
        self, db: SqlAlchemy, pm: ProviderManager, compressor: Compressor, interactions: list[Interaction]
    ) -> None:
        session = await _open(db, pm, RecordingOutbound())
        branch = await _started(session, db)

        await Compress(compressor).execute(session)

        assert db.summaries.get(branch) == Summary(covered=4, content="Hi")
        assert [(i.status, i.branch_id) for i in interactions] == [(InteractionStatus.OK, None)]

    async def test_cancel_records_the_call_interrupted_and_stores_nothing(
        self,
        db: SqlAlchemy,
        pm: ProviderManager,
        compressor: Compressor,
        interactions: list[Interaction],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        block, started = threading.Event(), threading.Event()
        chat_once = FakeAdapter.chat

        def chat_started(adapter: FakeAdapter, payload: dict, *, stream: bool) -> Iterator[ChatResponse]:
            assert stream
            started.set()
            return chat_once(adapter, payload, stream=True)

        monkeypatch.setattr(FakeAdapter, "block", block)
        monkeypatch.setattr(FakeAdapter, "chat", chat_started)
        monkeypatch.setattr(FakeAdapter, "replies", [chunk("Hi"), chunk("", done=True)])
        session = await _open(db, pm, RecordingOutbound())
        branch = await _started(session, db)
        registry = CommandRegistry()
        registry.set_commands([Compress(compressor)])

        session.start(registry.execute(session, "/compress"), "/compress")
        assert await asyncio.to_thread(started.wait, 5)
        stopping = asyncio.create_task(session.stop())
        await asyncio.sleep(0.05)
        block.set()
        await stopping

        assert db.summaries.get(branch) is None
        assert [i.status for i in interactions] == [InteractionStatus.INTERRUPTED]
        assert _systems(session) == []


class TestBuildCommands:
    def test_compress_is_registered_with_a_compressor(
        self, db: SqlAlchemy, pm: ProviderManager, compressor: Compressor
    ) -> None:
        registry = CommandRegistry()
        sessions = SessionRegistry(pm, PersonaManager(db), ToolManager())

        commands = build_commands(sessions, registry, db, pm, ToolManager(), compressor=compressor)

        assert any(isinstance(command, Compress) for command in commands)

    def test_compress_is_left_out_without_a_compressor(self, db: SqlAlchemy, pm: ProviderManager) -> None:
        registry = CommandRegistry()
        sessions = SessionRegistry(pm, PersonaManager(db), ToolManager())

        commands = build_commands(sessions, registry, db, pm, ToolManager())

        assert not any(isinstance(command, Compress) for command in commands)


@pytest.mark.usefixtures("replies")
class TestNearBudgetQuestion:
    @pytest.fixture
    def pipeline(self, pm: ProviderManager, compressor: Compressor, budget: ContextBudget) -> ChatPipeline:
        return ChatPipeline(ModelStage(ReAct(pm, ToolManager()))).use(CompressionStage(compressor, budget))

    async def test_new_conversation_saves_the_prompt_in_a_new_branch(
        self,
        db: SqlAlchemy,
        pm: ProviderManager,
        pipeline: ChatPipeline,
        events: EventBus,
        budget: ContextBudget,
    ) -> None:
        session = await _open(db, pm, AnsweringOutbound(CompressionChoice.NEW_CONVERSATION))
        old = await _started(session, db)
        old_rag = session.conv.rag_id
        _measure(events, old, NEAR)

        await chat(session, "hello", pipeline, db, events, budget=budget)

        new = session.conv.uuid
        assert new is not None
        assert new != old
        assert session.conv.rag_id != old_rag
        assert session.conv.model == MODEL
        assert [message.content for message in db.get_messages(new)] == ["hello", "Hi"]
        assert len(db.get_messages(old)) == 6
        assert db.summaries.get(old) is None
        kinds = [type(message) for message in _sent(session)]
        assert kinds[:3] == [AskCompression, History, ContextUsage]
        assert _sent(session)[1] == History(messages=[])
        assert _sent(session)[2] == ContextUsage(used=0, budget=BUDGET, warning_threshold=0.8)

    async def test_new_conversation_clears_the_cache(
        self,
        db: SqlAlchemy,
        pm: ProviderManager,
        pipeline: ChatPipeline,
        events: EventBus,
        budget: ContextBudget,
    ) -> None:
        session = await _open(db, pm, AnsweringOutbound(CompressionChoice.NEW_CONVERSATION))
        cache = FakeSessionCache(answers={"old": "answer"})
        session.cache = cast("SessionCache", cache)
        _measure(events, await _started(session, db), NEAR)

        await chat(session, "hello", pipeline, db, events, budget=budget)

        assert "old" not in cache.answers

    @pytest.mark.parametrize("answer", [CompressionChoice.COMPRESS, None])
    async def test_compress_or_no_answer_compresses_first(  # noqa: PLR0913, PLR0917 - the shared fixtures
        self,
        db: SqlAlchemy,
        pm: ProviderManager,
        pipeline: ChatPipeline,
        events: EventBus,
        budget: ContextBudget,
        interactions: list[Interaction],
        answer: CompressionChoice | None,
    ) -> None:
        session = await _open(db, pm, AnsweringOutbound(answer))
        branch = await _started(session, db)
        _measure(events, branch, NEAR)
        interactions.clear()

        await chat(session, "hello", pipeline, db, events, budget=budget)

        assert session.conv.uuid == branch
        assert db.summaries.get(branch) == Summary(covered=4, content="Hi")
        assert [i.branch_id for i in interactions] == [None, branch]
        assert any(text.startswith("Summarised the 4 oldest messages") for text in _systems(session))
        assert (AskCompression in [type(message) for message in _sent(session)]) == (answer is not None)

    async def test_no_question_below_the_threshold(
        self,
        db: SqlAlchemy,
        pm: ProviderManager,
        pipeline: ChatPipeline,
        events: EventBus,
        budget: ContextBudget,
    ) -> None:
        session = await _open(db, pm, AnsweringOutbound(CompressionChoice.NEW_CONVERSATION))
        branch = await _started(session, db)
        _measure(events, branch, FAR)

        await chat(session, "hello", pipeline, db, events, budget=budget)

        assert session.conv.uuid == branch
        assert not any(isinstance(message, AskCompression) for message in _sent(session))

    async def test_no_question_before_the_conversation_starts(
        self, db: SqlAlchemy, pm: ProviderManager, pipeline: ChatPipeline, events: EventBus, budget: ContextBudget
    ) -> None:
        session = await _open(db, pm, AnsweringOutbound(CompressionChoice.NEW_CONVERSATION))
        _measure(events, uuid4(), NEAR)

        await chat(session, "hello", pipeline, db, events, budget=budget)

        assert session.conv.uuid is not None
        assert not any(isinstance(message, AskCompression) for message in _sent(session))
