import asyncio
import logging
import threading
from collections.abc import Iterator
from contextlib import aclosing
from pathlib import Path

import pytest
from sensai_protocol import AiChunk
from sensai_server.chat import chat
from sensai_server.chat.react import ReAct
from sensai_server.events import EventBus
from sensai_server.llm.error import RequestCallError
from sensai_server.llm.persona import Persona, PersonaManager
from sensai_server.llm.provider import ChatResponse, ProviderManager
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai_server.monitoring.interaction import Interaction, InteractionStatus
from sensai_server.pipeline import ChatContext, ChatPipeline
from sensai_server.pipeline.modules import ModelStage, PersonaStage
from sensai_server.pipeline.modules.guardrails import BadWordsGuard
from sensai_server.session import Session
from sensai_server.tools.tool import Tool, ToolContext
from sensai_server.tools.tool_manager import ToolManager

from tests.server.conftest import MODEL, SETTINGS, db_url
from tests.server.fakes import FakeAdapter, RecordingOutbound, call, chunk, tool_call_chunk

TIMEOUT = 2
ANSWER = "Bonjour !"
THOUGHT = "I need a recent fact.\nTool needed: yes"
PERSONA = Persona(None, "analyst", "Analyses things", "You are an analyst.")


class SearchTool(Tool[str]):
    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search the web."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def run(self, ctx: ToolContext, **kwargs: object) -> str:
        del ctx, kwargs
        return "a result"


@pytest.fixture(autouse=True)
def answers() -> None:
    """Every call not scripted answers `ANSWER`, by a model able to call tools."""
    FakeAdapter.replies = [chunk(ANSWER, done=True)]
    FakeAdapter.tools = True


def _context(
    recorded: list[Interaction], manager: ToolManager | None = None, *, max_iterations: int = 0
) -> ChatContext:
    return ChatContext(
        message="Bonjour",
        out=RecordingOutbound(),
        model=MODEL,
        messages=[{"role": "user", "content": "Bonjour"}],
        tools=manager.new_state() if manager is not None else None,
        max_iterations=max_iterations,
        on_interaction=recorded.append,
    )


def _react(manager: ToolManager | None = None) -> ReAct:
    return ReAct(ProviderManager(SETTINGS.llm), manager or ToolManager())


async def _run(context: ChatContext, manager: ToolManager | None = None) -> list[ChatResponse]:
    async with aclosing(_react(manager).generate_answer(context)) as chunks:
        return [item async for item in chunks]


def _fail_on_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_chat(_self: FakeAdapter, _payload: dict, *, stream: bool) -> Iterator:  # noqa: ARG001
        msg = "connection refused"
        raise RequestCallError(msg)

    monkeypatch.setattr(FakeAdapter, "chat", failing_chat)


class TestReActRecording:
    async def test_plain_answer_records_one_ok_call(self) -> None:
        recorded: list[Interaction] = []
        await _run(_context(recorded))

        [interaction] = recorded
        assert interaction.status is InteractionStatus.OK
        assert interaction.model == MODEL
        assert interaction.response == ANSWER
        assert interaction.error is None
        assert interaction.request == FakeAdapter.payloads[0]["messages"]

    async def test_exchange_with_a_tool_records_every_call(self) -> None:
        FakeAdapter.script = [[chunk(THOUGHT, done=True)], [tool_call_chunk(call("search")), chunk("", done=True)]]
        manager = ToolManager([SearchTool()])
        recorded: list[Interaction] = []

        await _run(_context(recorded, manager, max_iterations=1), manager)

        assert len(recorded) == len(FakeAdapter.payloads) == 3
        assert [interaction.status for interaction in recorded] == [InteractionStatus.OK] * 3
        assert [interaction.response for interaction in recorded] == [THOUGHT, None, ANSWER]
        assert recorded[1].tool_calls == [call("search")]

    async def test_error_chunk_records_an_error(self) -> None:
        FakeAdapter.replies = [{"error": "boom"}]
        recorded: list[Interaction] = []

        chunks = await _run(_context(recorded))

        assert [item.error for item in chunks] == ["boom"]
        [interaction] = recorded
        assert interaction.status is InteractionStatus.ERROR
        assert interaction.error == "boom"

    async def test_provider_exception_records_an_error_and_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fail_on_chat(monkeypatch)
        recorded: list[Interaction] = []

        with pytest.raises(RequestCallError):
            await _run(_context(recorded))

        [interaction] = recorded
        assert interaction.status is InteractionStatus.ERROR
        assert interaction.error is not None
        assert "connection refused" in interaction.error

    async def test_closing_early_records_an_interruption(self) -> None:
        FakeAdapter.replies = [chunk("a"), chunk("b", done=True)]
        recorded: list[Interaction] = []

        async with aclosing(_react().generate_answer(_context(recorded))) as chunks:
            async for _ in chunks:
                break

        [interaction] = recorded
        assert interaction.status is InteractionStatus.INTERRUPTED
        assert interaction.response == "a"

    async def test_cancelling_records_an_interruption_and_still_cancels(self) -> None:
        block = threading.Event()
        FakeAdapter.block = block
        FakeAdapter.replies = [chunk("a"), chunk("b", done=True)]
        recorded: list[Interaction] = []
        first = asyncio.Event()

        async def consume() -> None:
            async with aclosing(_react().generate_answer(_context(recorded))) as chunks:
                async for _ in chunks:
                    first.set()

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(first.wait(), TIMEOUT)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            block.set()

        [interaction] = recorded
        assert interaction.status is InteractionStatus.INTERRUPTED
        assert await asyncio.to_thread(FakeAdapter.closed.wait, TIMEOUT)


@pytest.fixture
def db(tmp_path: Path) -> SqlAlchemy:
    return SqlAlchemy(db_url(tmp_path))


@pytest.fixture
def session(persona_manager: PersonaManager) -> Session:
    session = Session(RecordingOutbound(), ProviderManager(SETTINGS.llm), persona_manager, ToolManager())
    session.conv.model = MODEL
    session.effort = "off"
    return session


@pytest.fixture
def pipeline() -> ChatPipeline:
    react = ReAct(ProviderManager(SETTINGS.llm), ToolManager())
    return ChatPipeline(ModelStage(react)).use(BadWordsGuard()).use(PersonaStage())


@pytest.fixture
def events() -> tuple[EventBus, list[Interaction]]:
    bus = EventBus()
    received: list[Interaction] = []
    bus.subscribe(Interaction, received.append)
    return bus, received


def _ai_chunks(session: Session) -> list[AiChunk]:
    assert isinstance(session.out, RecordingOutbound)
    return [message for message in session.out.sent if isinstance(message, AiChunk)]


class TestChatEmission:
    async def test_emits_the_call_tagged_with_its_branch_and_persona(
        self,
        session: Session,
        pipeline: ChatPipeline,
        db: SqlAlchemy,
        events: tuple[EventBus, list[Interaction]],
    ) -> None:
        bus, received = events
        session.conv.persona = PERSONA

        await chat(session, "hello", pipeline, db, bus)

        [interaction] = received
        assert interaction.status is InteractionStatus.OK
        assert interaction.branch_id is not None
        assert interaction.branch_id == session.conv.uuid
        assert interaction.persona == PERSONA.name

    async def test_no_persona_is_tagged_none(
        self,
        session: Session,
        pipeline: ChatPipeline,
        db: SqlAlchemy,
        events: tuple[EventBus, list[Interaction]],
    ) -> None:
        bus, received = events
        await chat(session, "hello", pipeline, db, bus)
        assert [interaction.persona for interaction in received] == [None]

    async def test_provider_exception_emits_an_error(
        self,
        session: Session,
        pipeline: ChatPipeline,
        db: SqlAlchemy,
        events: tuple[EventBus, list[Interaction]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bus, received = events
        _fail_on_chat(monkeypatch)
        await chat(session, "hello", pipeline, db, bus)
        assert [interaction.status for interaction in received] == [InteractionStatus.ERROR]

    async def test_answer_without_the_model_emits_nothing(
        self,
        session: Session,
        pipeline: ChatPipeline,
        db: SqlAlchemy,
        events: tuple[EventBus, list[Interaction]],
    ) -> None:
        bus, received = events
        await chat(session, "fuck", pipeline, db, bus)
        assert _ai_chunks(session)
        assert received == []

    async def test_raising_subscriber_leaves_the_answer_intact(
        self,
        session: Session,
        pipeline: ChatPipeline,
        db: SqlAlchemy,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def broken(_: Interaction) -> None:
            msg = "subscriber down"
            raise RuntimeError(msg)

        bus = EventBus()
        bus.subscribe(Interaction, broken)
        FakeAdapter.replies = [chunk("hi "), chunk("there", done=True)]

        with caplog.at_level(logging.ERROR):
            await chat(session, "hello", pipeline, db, bus)

        assert [item.content for item in _ai_chunks(session)] == ["hi ", "there"]
        assert session.conv.history.to_json()[-1] == {"role": "assistant", "content": "hi there"}
        assert "subscriber down" in caplog.text
