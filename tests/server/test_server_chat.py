import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from sensai_protocol import AiChunk, Error, Input, Role, ServerMessage, System
from sensai_server.chat import chat
from sensai_server.chat.chat import BRANCH_NAME_LENGTH, _to_ai_chunk
from sensai_server.chat.react import ANSWER_INSTRUCTION, EFFORTS, ReAct
from sensai_server.llm.error import RequestCallError
from sensai_server.llm.persona import Persona, PersonaManager
from sensai_server.llm.provider import ChatResponse, ProviderManager
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai_server.pipeline import ChatPipeline
from sensai_server.pipeline.modules import ModelStage, PersonaStage
from sensai_server.pipeline.modules.guardrails import WARNING, BadWordsGuard
from sensai_server.session import Session
from sensai_server.tools.tool_manager import ToolManager

from tests.server.conftest import MODEL, SETTINGS, db_url
from tests.server.fakes import FakeAdapter, RecordingNext, RecordingOutbound, chunk

PERSONA = Persona(None, "analyst", "Analyses things", "You are an analyst.")
TIMEOUT = 2


@pytest.fixture
def db(tmp_path: Path) -> SqlAlchemy:
    return SqlAlchemy(db_url(tmp_path))


@pytest.fixture
def pm() -> ProviderManager:
    return ProviderManager(SETTINGS.llm)


@pytest.fixture
def session(pm: ProviderManager, persona_manager: PersonaManager) -> Session:
    session = Session(RecordingOutbound(), pm, persona_manager, ToolManager())
    session.conv.model = MODEL
    # `FakeAdapter` answers every call with the same stream: the pipeline makes a single answering call.
    session.effort = "off"
    return session


@pytest.fixture
def pipeline(pm: ProviderManager) -> ChatPipeline:
    """The pipeline `create_app` builds, on `FakeAdapter`."""
    return ChatPipeline(ModelStage(ReAct(pm, ToolManager()))).use(BadWordsGuard()).use(PersonaStage())


class _SignallingOutbound(RecordingOutbound):
    """A `RecordingOutbound` that sets `chunk_sent` once it has sent an `ai_chunk`."""

    def __init__(self) -> None:
        super().__init__()
        self.chunk_sent = asyncio.Event()

    async def send(self, message: ServerMessage) -> bool:
        await super().send(message)
        if isinstance(message, AiChunk):
            self.chunk_sent.set()
        return True


def _sent(session: Session) -> list:
    assert isinstance(session.out, RecordingOutbound)
    return session.out.sent


def _history(session: Session) -> list[dict]:
    return session.conv.history.to_json()


class TestChat:
    async def test_does_not_echo_the_prompt(self, session: Session, pipeline: ChatPipeline, db: SqlAlchemy) -> None:
        FakeAdapter.replies = [chunk("hi", done=True)]
        await chat(session, "hello", pipeline, db)
        assert not any(isinstance(message, Input) for message in _sent(session))

    async def test_streams_chunks_in_order(self, session: Session, pipeline: ChatPipeline, db: SqlAlchemy) -> None:
        FakeAdapter.replies = [chunk("a"), chunk("b"), chunk("c", done=True)]
        await chat(session, "hello", pipeline, db)
        chunks = [message for message in _sent(session) if isinstance(message, AiChunk)]
        assert [item.content for item in chunks] == ["a", "b", "c"]
        assert [item.done for item in chunks] == [False, False, True]

    async def test_saves_both_messages(self, session: Session, pipeline: ChatPipeline, db: SqlAlchemy) -> None:
        FakeAdapter.replies = [chunk("hi "), chunk("there", done=True)]
        await chat(session, "hello", pipeline, db)
        expected = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"}]
        assert _history(session) == expected
        assert session.conv.uuid is not None
        stored = db.get_messages(session.conv.uuid)
        assert [(message.role, message.content) for message in stored] == [
            (Role.USER, "hello"),
            (Role.ASSISTANT, "hi there"),
        ]

    async def test_creates_the_branch_once(self, session: Session, pipeline: ChatPipeline, db: SqlAlchemy) -> None:
        prompt = "x" * (BRANCH_NAME_LENGTH + 10)
        FakeAdapter.replies = [chunk("a", done=True)]
        await chat(session, prompt, pipeline, db)
        branch = session.conv.uuid
        assert branch is not None
        assert db.get_branches()[branch] == prompt[:BRANCH_NAME_LENGTH]
        await chat(session, "again", pipeline, db)
        assert session.conv.uuid == branch
        assert list(db.get_branches()) == [branch]

    async def test_provider_error(self, session: Session, db: SqlAlchemy) -> None:
        pipeline = ChatPipeline(RecordingNext(ChatResponse({"error": "boom"}), ChatResponse(chunk("never"))))
        await chat(session, "hello", pipeline, db)
        chunks = [message for message in _sent(session) if isinstance(message, AiChunk)]
        assert chunks == [AiChunk(error="boom", done=True)]
        assert _history(session) == [{"role": "user", "content": "hello"}]

    async def test_error_inside_the_pipeline_sends_an_error(
        self, session: Session, pipeline: ChatPipeline, db: SqlAlchemy
    ) -> None:
        session.conv.model = "nope"
        await chat(session, "hello", pipeline, db)
        errors = [message for message in _sent(session) if isinstance(message, Error)]
        assert [error.message for error in errors] == ["No provider serves model nope"]
        assert not any(isinstance(message, AiChunk) for message in _sent(session))
        assert _history(session) == [{"role": "user", "content": "hello"}]

    async def test_provider_exception_sends_an_error(
        self, session: Session, pipeline: ChatPipeline, db: SqlAlchemy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing_chat(_self: FakeAdapter, _payload: dict, *, stream: bool) -> Iterator:  # noqa: ARG001
            msg = "connection refused"
            raise RequestCallError(msg)

        monkeypatch.setattr(FakeAdapter, "chat", failing_chat)
        await chat(session, "hello", pipeline, db)
        assert any(isinstance(message, Error) for message in _sent(session))
        assert _history(session) == [{"role": "user", "content": "hello"}]

    async def test_blank_answer_is_not_saved(self, session: Session, pipeline: ChatPipeline, db: SqlAlchemy) -> None:
        FakeAdapter.replies = [chunk("  ", done=True)]
        await chat(session, "hello", pipeline, db)
        assert [message for message in _sent(session) if isinstance(message, AiChunk)] == [
            AiChunk(content="  ", done=True)
        ]
        assert _history(session) == [{"role": "user", "content": "hello"}]
        assert session.conv.uuid is not None
        assert len(db.get_messages(session.conv.uuid)) == 1

    async def test_failed_user_save_stops_the_chat(
        self, session: Session, pipeline: ChatPipeline, db: SqlAlchemy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(db, "add_message_to_branch", lambda *_: None)
        FakeAdapter.replies = [chunk("a", done=True)]
        await chat(session, "hello", pipeline, db)
        assert [type(message) for message in _sent(session)] == [System]
        assert FakeAdapter.last_payload is None
        assert _history(session) == []

    async def test_second_chat_sends_the_whole_history(
        self, session: Session, pipeline: ChatPipeline, db: SqlAlchemy
    ) -> None:
        FakeAdapter.replies = [chunk("hi", done=True)]
        await chat(session, "hello", pipeline, db)
        await chat(session, "again", pipeline, db)
        assert FakeAdapter.last_payload is not None
        assert FakeAdapter.last_payload["messages"] == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "again"},
            {"role": "user", "content": ANSWER_INSTRUCTION},
        ]

    async def test_no_model_sends_an_error(self, session: Session, pipeline: ChatPipeline, db: SqlAlchemy) -> None:
        session.conv.model = None
        await chat(session, "hello", pipeline, db)
        assert [type(message) for message in _sent(session)] == [Error]
        assert session.conv.uuid is None

    async def test_bad_word_is_answered_by_the_guard(
        self, session: Session, pipeline: ChatPipeline, db: SqlAlchemy
    ) -> None:
        await chat(session, "fuck", pipeline, db)
        assert [message for message in _sent(session) if isinstance(message, AiChunk)] == [
            AiChunk(content=WARNING, done=True)
        ]
        assert FakeAdapter.last_payload is None
        assert _history(session) == [
            {"role": "user", "content": "fuck"},
            {"role": "assistant", "content": WARNING},
        ]

    async def test_persona_in_payload_not_in_history(
        self, session: Session, pipeline: ChatPipeline, db: SqlAlchemy
    ) -> None:
        session.conv.persona = PERSONA
        FakeAdapter.replies = [chunk("a", done=True)]
        await chat(session, "hello", pipeline, db)
        assert FakeAdapter.last_payload is not None
        assert FakeAdapter.last_payload["messages"] == [
            {"role": "system", "content": PERSONA.prompt},
            {"role": "user", "content": "hello"},
            {"role": "user", "content": ANSWER_INSTRUCTION},
        ]
        assert all(message["role"] != "system" for message in _history(session))

    async def test_context_carries_the_tool_state_of_the_session(self, session: Session, db: SqlAlchemy) -> None:
        model = RecordingNext(ChatResponse(chunk("a", done=True)))
        await chat(session, "hello", ChatPipeline(model), db)
        assert [context.tools for context in model.calls] == [session.tools]
        assert model.calls[0].tools is session.tools

    @pytest.mark.parametrize(("level", "steps"), list(EFFORTS.items()))
    async def test_context_carries_the_effort_of_the_session(
        self, session: Session, db: SqlAlchemy, level: str, steps: int
    ) -> None:
        session.effort = level
        model = RecordingNext(ChatResponse(chunk("a", done=True)))
        await chat(session, "hello", ChatPipeline(model), db)
        assert [context.max_iterations for context in model.calls] == [steps]

    async def test_cancel_saves_no_answer(self, session: Session, pipeline: ChatPipeline, db: SqlAlchemy) -> None:
        block = threading.Event()
        FakeAdapter.block = block
        FakeAdapter.replies = [chunk("a"), chunk("b", done=True)]
        out = _SignallingOutbound()
        session.out = out
        task = asyncio.create_task(chat(session, "hello", pipeline, db))
        try:
            await asyncio.wait_for(out.chunk_sent.wait(), TIMEOUT)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            block.set()
        assert await asyncio.to_thread(FakeAdapter.closed.wait, TIMEOUT)
        assert _history(session) == [{"role": "user", "content": "hello"}]


class TestToAiChunk:
    def test_maps_fields(self) -> None:
        response = ChatResponse({"done": False, "message": {"role": "assistant", "content": "a", "thinking": "t"}})
        assert _to_ai_chunk(response) == AiChunk(content="a", thinking="t")

    def test_empty_tool_calls_are_none(self) -> None:
        response = ChatResponse({"done": True, "message": {"role": "assistant", "content": "a", "tool_calls": []}})
        assert _to_ai_chunk(response).tool_calls is None

    def test_error_is_done(self) -> None:
        assert _to_ai_chunk(ChatResponse({"error": "boom"})) == AiChunk(error="boom", done=True)
