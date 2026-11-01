from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest
from sensai_protocol import AiChunk, ContextUsage, History, Login, ServerMessage, Session, System
from sensai_server.app import _handlers as handlers
from sensai_server.chat import chat
from sensai_server.chat.react import ReAct
from sensai_server.chat.usage import send_context_usage
from sensai_server.commands._branch import Branch
from sensai_server.commands._model import Model
from sensai_server.config.settings import ContextSettings
from sensai_server.events import EventBus
from sensai_server.llm.persona import PersonaManager
from sensai_server.llm.provider import ProviderManager
from sensai_server.llm.responses import ChatMetrics
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai_server.monitoring.budget import ContextBudget
from sensai_server.monitoring.interaction import Interaction, InteractionStatus
from sensai_server.pipeline import ChatPipeline
from sensai_server.pipeline.modules import ModelStage
from sensai_server.session import Session as ServerSession
from sensai_server.session import SessionRegistry
from sensai_server.tools.tool_manager import ToolManager

from tests.server.conftest import MODEL, SETTINGS, db_url
from tests.server.fakes import FakeAdapter, RecordingOutbound, chunk

OTHER_MODEL = "b"
LENGTHS = {MODEL: 4096, OTHER_MODEL: 8192}
THRESHOLD = 0.8


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
def budget(events: EventBus) -> ContextBudget:
    """A budget measuring the calls emitted on `events`, against the context lengths of `LENGTHS`."""
    budget = ContextBudget(ContextSettings(warning_threshold=THRESHOLD), LENGTHS.get)
    budget.attach(events)
    return budget


@pytest.fixture
async def session(db: SqlAlchemy, pm: ProviderManager) -> ServerSession:
    registry = SessionRegistry(pm, PersonaManager(db), ToolManager())
    opened = await registry.open(RecordingOutbound(), MODEL)
    assert opened is not None
    opened.effort = "off"
    _sent(opened).clear()
    return opened


def _sent(session: ServerSession) -> list[ServerMessage]:
    return cast("RecordingOutbound", session.out).sent


def _usages(session: ServerSession) -> list[ContextUsage]:
    return [message for message in _sent(session) if isinstance(message, ContextUsage)]


def _usage(used: int | None, budget: int | None) -> ContextUsage:
    return ContextUsage(used=used, budget=budget, warning_threshold=THRESHOLD)


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


def _new_branch(db: SqlAlchemy) -> UUID:
    created = db.create_branch(model=MODEL)
    assert created is not None
    return created


class TestSendContextUsage:
    async def test_sends_the_usage_of_the_current_branch(
        self, session: ServerSession, budget: ContextBudget, events: EventBus
    ) -> None:
        session.conv.uuid = uuid4()
        _measure(events, session.conv.uuid, 1000)

        await send_context_usage(session, budget)

        assert _sent(session) == [_usage(1000, 4096)]

    async def test_sends_nothing_without_a_budget(self, session: ServerSession) -> None:
        await send_context_usage(session, None)

        assert _sent(session) == []

    async def test_sends_nothing_without_a_model(self, session: ServerSession, budget: ContextBudget) -> None:
        session.conv.model = None

        await send_context_usage(session, budget)

        assert _sent(session) == []


class TestLogin:
    async def test_a_new_conversation_takes_nothing(
        self, db: SqlAlchemy, pm: ProviderManager, budget: ContextBudget
    ) -> None:
        out = RecordingOutbound()
        registry = SessionRegistry(pm, PersonaManager(db), ToolManager())

        opened = await handlers.handle_login(registry, out, None, Login(model=MODEL), [], budget=budget)

        assert opened is not None
        assert [type(message) for message in out.sent] == [Session, ContextUsage]
        assert out.sent[1] == _usage(0, 4096)

    async def test_an_unknown_model_sends_no_usage(
        self, db: SqlAlchemy, pm: ProviderManager, budget: ContextBudget
    ) -> None:
        out = RecordingOutbound()
        registry = SessionRegistry(pm, PersonaManager(db), ToolManager())

        await handlers.handle_login(registry, out, None, Login(model="unknown"), [], budget=budget)

        assert not any(isinstance(message, ContextUsage) for message in out.sent)


class TestChat:
    @pytest.fixture
    def pipeline(self, pm: ProviderManager) -> ChatPipeline:
        return ChatPipeline(ModelStage(ReAct(pm, ToolManager())))

    @pytest.fixture
    def measured_replies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Answer with a stream whose last chunk reports 100 prompt tokens and 20 completion tokens."""
        done = chunk("", done=True) | {"prompt_eval_count": 100, "eval_count": 20}
        monkeypatch.setattr(FakeAdapter, "replies", [chunk("Hi"), done])

    @pytest.mark.usefixtures("measured_replies")
    async def test_sends_the_measure_of_the_last_call_after_the_answer(
        self, session: ServerSession, pipeline: ChatPipeline, db: SqlAlchemy, events: EventBus, budget: ContextBudget
    ) -> None:
        await chat(session, "hello", pipeline, db, events, budget=budget)

        assert _usages(session) == [_usage(120, 4096)]
        assert isinstance(_sent(session)[-2], AiChunk)
        assert _sent(session)[-1] == _usage(120, 4096)

    async def test_a_chat_that_cannot_start_sends_no_usage(
        self, session: ServerSession, pipeline: ChatPipeline, db: SqlAlchemy, events: EventBus, budget: ContextBudget
    ) -> None:
        session.conv.model = None

        await chat(session, "hello", pipeline, db, events, budget=budget)

        assert _usages(session) == []


class TestModelCommand:
    async def test_same_usage_against_the_new_budget(
        self, session: ServerSession, budget: ContextBudget, events: EventBus
    ) -> None:
        session.conv.uuid = uuid4()
        _measure(events, session.conv.uuid, 1000)

        await Model(budget).execute(session, OTHER_MODEL)

        assert _usages(session) == [_usage(1000, 8192)]

    async def test_an_unknown_model_sends_no_usage(self, session: ServerSession, budget: ContextBudget) -> None:
        await Model(budget).execute(session, "unknown")

        assert _usages(session) == []

    async def test_listing_the_models_sends_no_usage(self, session: ServerSession, budget: ContextBudget) -> None:
        await Model(budget).execute(session)

        assert _usages(session) == []


class TestBranchCommand:
    async def test_set_sends_the_usage_of_the_loaded_branch(
        self, session: ServerSession, db: SqlAlchemy, pm: ProviderManager, budget: ContextBudget, events: EventBus
    ) -> None:
        target = _new_branch(db)
        _measure(events, target, 1500)

        await Branch(db, pm, budget).execute(session, "set", str(target))

        assert [type(message) for message in _sent(session)] == [History, ContextUsage]
        assert _sent(session)[-1] == _usage(1500, 4096)

    async def test_set_on_a_branch_never_measured_has_an_unknown_usage(
        self, session: ServerSession, db: SqlAlchemy, pm: ProviderManager, budget: ContextBudget
    ) -> None:
        await Branch(db, pm, budget).execute(session, "set", str(_new_branch(db)))

        assert _usages(session) == [_usage(None, 4096)]

    async def test_fork_sends_the_usage_of_the_new_branch(
        self, session: ServerSession, db: SqlAlchemy, pm: ProviderManager, budget: ContextBudget, events: EventBus
    ) -> None:
        source = _new_branch(db)
        _measure(events, source, 1500)
        branch = Branch(db, pm, budget)
        await branch.execute(session, "set", str(source))
        _sent(session).clear()

        await branch.execute(session, "fork")

        assert session.conv.uuid != source
        assert _usages(session) == [_usage(None, 4096)]

    @pytest.mark.parametrize("target", ["not-a-uuid", str(uuid4())])
    async def test_a_failed_set_sends_no_usage(
        self, session: ServerSession, db: SqlAlchemy, pm: ProviderManager, budget: ContextBudget, target: str
    ) -> None:
        await Branch(db, pm, budget).execute(session, "set", target)

        assert _usages(session) == []
        assert isinstance(_sent(session)[-1], System)
