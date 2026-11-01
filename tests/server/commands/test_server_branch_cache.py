from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest
from sensai_protocol import System
from sensai_protocol.server_messages import History, HistoryMessage
from sensai_server.cache import SessionCache
from sensai_server.commands._branch import Branch
from sensai_server.llm.persona import PersonaManager
from sensai_server.llm.provider import ProviderManager
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai_server.session import Session, SessionRegistry
from sensai_server.tools.tool_manager import ToolManager

from tests.server.conftest import MODEL, SETTINGS, db_url
from tests.server.fakes import FakeSessionCache, RecordingOutbound


@pytest.fixture
def db(tmp_path: Path) -> SqlAlchemy:
    return SqlAlchemy(db_url(tmp_path))


@pytest.fixture
def pm() -> ProviderManager:
    return ProviderManager(SETTINGS.llm)


@pytest.fixture
def branch(db: SqlAlchemy, pm: ProviderManager) -> Branch:
    return Branch(db, pm)


@pytest.fixture
def cache() -> FakeSessionCache:
    return FakeSessionCache()


@pytest.fixture
async def session(db: SqlAlchemy, pm: ProviderManager, cache: FakeSessionCache) -> Session:
    """A session whose semantic cache is `cache`."""
    registry = SessionRegistry(pm, PersonaManager(db), ToolManager())
    opened = await registry.open(RecordingOutbound(), MODEL)
    assert opened is not None
    opened.cache = cast("SessionCache", cache)
    return opened


def _new_branch(db: SqlAlchemy) -> UUID:
    created = db.create_branch(model=MODEL)
    assert created is not None
    return created


def _sent(session: Session) -> list:
    return cast("RecordingOutbound", session.out).sent


class TestBranchClearsTheCache:
    async def test_set_clears_the_cache(
        self, branch: Branch, session: Session, db: SqlAlchemy, cache: FakeSessionCache
    ) -> None:
        target = _new_branch(db)

        await branch.execute(session, "set", str(target))

        assert session.conv.uuid == target
        assert cache.clears == 1

    async def test_fork_clears_the_cache(
        self, branch: Branch, session: Session, db: SqlAlchemy, cache: FakeSessionCache
    ) -> None:
        start = _new_branch(db)
        await branch.execute(session, "set", str(start))
        cache.clears = 0

        await branch.execute(session, "fork")

        assert session.conv.uuid not in {None, start}
        assert cache.clears == 1

    async def test_the_cache_is_cleared_before_the_history_is_sent(
        self, branch: Branch, session: Session, db: SqlAlchemy, cache: FakeSessionCache
    ) -> None:
        clears_at_history: list[int] = []

        class Watching(RecordingOutbound):
            async def load_conversation(self, messages: list[HistoryMessage]) -> None:
                clears_at_history.append(cache.clears)
                await super().load_conversation(messages)

        session.out = Watching()
        await branch.execute(session, "set", str(_new_branch(db)))

        assert clears_at_history == [1]


class TestBranchKeepsTheCache:
    async def test_an_unknown_branch_keeps_it(self, branch: Branch, session: Session, cache: FakeSessionCache) -> None:
        await branch.execute(session, "set", str(uuid4()))
        assert cache.clears == 0

    async def test_an_invalid_uuid_keeps_it(self, branch: Branch, session: Session, cache: FakeSessionCache) -> None:
        await branch.execute(session, "set", "not-a-uuid")
        assert cache.clears == 0

    async def test_the_current_branch_keeps_it(
        self, branch: Branch, session: Session, db: SqlAlchemy, cache: FakeSessionCache
    ) -> None:
        target = _new_branch(db)
        await branch.execute(session, "set", str(target))
        cache.clears = 0

        await branch.execute(session, "set", str(target))

        assert cache.clears == 0

    async def test_no_cache_still_switches(self, branch: Branch, session: Session, db: SqlAlchemy) -> None:
        session.cache = None
        target = _new_branch(db)

        await branch.execute(session, "set", str(target))

        assert session.conv.uuid == target


class TestFailingClear:
    async def test_a_failing_clear_is_reported_and_the_switch_completes(
        self, branch: Branch, session: Session, db: SqlAlchemy, cache: FakeSessionCache
    ) -> None:
        cache.failing_clear = True
        target = _new_branch(db)

        await branch.execute(session, "set", str(target))

        assert session.conv.uuid == target
        sent = _sent(session)
        assert any(
            isinstance(message, System) and message.message.startswith("The semantic cache could not be cleared:")
            for message in sent
        )
        assert isinstance(sent[-1], History)
