from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest
from sensai_protocol import System
from sensai_protocol.server_messages import History, Role
from sensai_server.commands._branch import Branch
from sensai_server.llm.persona import PersonaManager
from sensai_server.llm.provider import ProviderManager
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai_server.session import Session, SessionRegistry
from sensai_server.tools.tool_manager import ToolManager

from tests.server.conftest import MODEL, SETTINGS, db_url
from tests.server.fakes import RecordingOutbound


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
async def session(db: SqlAlchemy, pm: ProviderManager) -> Session:
    registry = SessionRegistry(pm, PersonaManager(db), ToolManager())
    opened = await registry.open(RecordingOutbound(), MODEL)
    assert opened is not None
    _sent(opened).clear()
    return opened


def _new_branch(db: SqlAlchemy, *contents: str) -> UUID:
    created = db.create_branch(model=MODEL)
    assert created is not None
    for i, content in enumerate(contents):
        db.add_message_to_branch(created, Role.USER if i % 2 == 0 else Role.ASSISTANT, content)
    return created


def _sent(session: Session) -> list:
    return cast("RecordingOutbound", session.out).sent


def _system(session: Session) -> list[str]:
    return [message.message for message in _sent(session) if isinstance(message, System)]


class TestDispatch:
    async def test_no_arguments_reports_missing_subcommand(self, branch: Branch, session: Session) -> None:
        await branch.execute(session)

        assert _system(session) == [f"Missing subcommand. Usage: {Branch.usage}"]

    async def test_unknown_subcommand_is_reported(self, branch: Branch, session: Session) -> None:
        await branch.execute(session, "nope")

        assert _system(session) == [f"Unknown subcommand 'nope'. Usage: {Branch.usage}"]

    @pytest.mark.parametrize("args", [("set",), ("remove", "a", "b"), ("rename", "a"), ("list", "extra")])
    async def test_wrong_number_of_arguments_reports_usage(
        self, branch: Branch, session: Session, args: tuple[str, ...]
    ) -> None:
        await branch.execute(session, *args)

        assert _system(session) == [f"Invalid arguments for '{args[0]}'. Usage: {Branch.usage}"]

    def test_completions_are_the_sorted_subcommands(self, branch: Branch, session: Session) -> None:
        assert branch.completions(session, []) == ["fork", "list", "remove", "rename", "set"]


class TestCompletions:
    @pytest.mark.parametrize("subcommand", ["set", "remove", "rename"])
    def test_uuid_subcommands_complete_with_branch_ids(
        self, branch: Branch, session: Session, db: SqlAlchemy, subcommand: str
    ) -> None:
        first, second = _new_branch(db), _new_branch(db)

        assert sorted(branch.completions(session, [subcommand])) == sorted([str(first), str(second)])

    @pytest.mark.parametrize("subcommand", ["list", "fork", "nope"])
    def test_other_subcommands_have_no_completion(
        self, branch: Branch, session: Session, db: SqlAlchemy, subcommand: str
    ) -> None:
        _new_branch(db)

        assert branch.completions(session, [subcommand]) == []

    def test_words_after_the_uuid_have_no_completion(self, branch: Branch, session: Session, db: SqlAlchemy) -> None:
        target = _new_branch(db)

        assert branch.completions(session, ["rename", str(target)]) == []

    def test_without_branches_returns_nothing(self, branch: Branch, session: Session) -> None:
        assert branch.completions(session, ["set"]) == []


class TestList:
    async def test_displays_each_branch_with_its_uuid(self, branch: Branch, session: Session, db: SqlAlchemy) -> None:
        first, second = _new_branch(db), _new_branch(db)

        await branch.execute(session, "list")

        sent = [message for message in _sent(session) if isinstance(message, System)]
        assert [message.message for message in sent] == [f"{first}: Branch", f"{second}: Branch 2"]
        assert [message.append for message in sent] == [False, True]

    async def test_without_branches_displays_nothing(self, branch: Branch, session: Session) -> None:
        await branch.execute(session, "list")

        assert _sent(session) == []


class TestSet:
    async def test_loads_the_branch_into_the_conversation(
        self, branch: Branch, session: Session, db: SqlAlchemy
    ) -> None:
        target = _new_branch(db, "hi", "hello")

        await branch.execute(session, "set", str(target))

        assert session.conv.uuid == target
        assert [(m.role, m.content) for m in session.conv.history.messages()] == [
            (Role.USER, "hi"),
            (Role.ASSISTANT, "hello"),
        ]

    async def test_sends_the_loaded_history_to_the_client(
        self, branch: Branch, session: Session, db: SqlAlchemy
    ) -> None:
        target = _new_branch(db, "hi", "hello")

        await branch.execute(session, "set", str(target))

        [history] = [message for message in _sent(session) if isinstance(message, History)]
        assert [(m.role, m.content) for m in history.messages] == [(Role.USER, "hi"), (Role.ASSISTANT, "hello")]
        assert [m.uuid for m in history.messages] == [m.uuid for m in db.get_messages(target)]

    async def test_invalid_uuid_is_reported(self, branch: Branch, session: Session) -> None:
        await branch.execute(session, "set", "not-a-uuid")

        assert _system(session) == ["parameter is not a valid UUID."]
        assert session.conv.uuid is None

    async def test_unknown_branch_is_reported_and_nothing_is_loaded(
        self, branch: Branch, session: Session, db: SqlAlchemy
    ) -> None:
        current = _new_branch(db, "kept")
        await branch.execute(session, "set", str(current))
        _sent(session).clear()

        await branch.execute(session, "set", str(uuid4()))

        assert len(_system(session)) == 1
        assert not any(isinstance(message, History) for message in _sent(session))
        assert session.conv.uuid == current
        assert [m.content for m in session.conv.history.messages()] == ["kept"]


class TestRemove:
    async def test_removes_the_branch(self, branch: Branch, session: Session, db: SqlAlchemy) -> None:
        target = _new_branch(db)

        await branch.execute(session, "remove", str(target))

        assert db.get_branches() == {}
        assert _system(session) == [f"Branch {target} removed"]

    async def test_invalid_uuid_is_reported(self, branch: Branch, session: Session, db: SqlAlchemy) -> None:
        _new_branch(db)

        await branch.execute(session, "remove", "not-a-uuid")

        assert _system(session) == ["parameter is not a valid UUID."]
        assert len(db.get_branches()) == 1

    async def test_current_branch_cannot_be_removed(self, branch: Branch, session: Session, db: SqlAlchemy) -> None:
        current = _new_branch(db)
        await branch.execute(session, "set", str(current))
        _sent(session).clear()

        await branch.execute(session, "remove", str(current))

        assert _system(session) == ["You cannot remove your current branch."]
        assert current in db.get_branches()

    async def test_unknown_branch_is_reported(self, branch: Branch, session: Session) -> None:
        target = uuid4()

        await branch.execute(session, "remove", str(target))

        assert _system(session) == [f"Failed to remove {target}"]


class TestRename:
    async def test_renames_the_branch(self, branch: Branch, session: Session, db: SqlAlchemy) -> None:
        target = _new_branch(db)

        await branch.execute(session, "rename", str(target), "travel")

        assert db.get_branches() == {target: "travel"}
        assert _system(session) == [f"Branch {target} renamed travel"]

    async def test_invalid_uuid_is_reported(self, branch: Branch, session: Session) -> None:
        await branch.execute(session, "rename", "not-a-uuid", "travel")

        assert _system(session) == ["parameter is not a valid UUID."]

    async def test_unknown_branch_is_reported(self, branch: Branch, session: Session, db: SqlAlchemy) -> None:
        kept = _new_branch(db)
        target = uuid4()

        await branch.execute(session, "rename", str(target), "travel")

        assert _system(session) == [f"Failed to rename {target}"]
        assert db.get_branches() == {kept: "Branch"}


class TestFork:
    async def test_creates_a_copy_of_the_current_branch_and_loads_it(
        self, branch: Branch, session: Session, db: SqlAlchemy
    ) -> None:
        start = _new_branch(db, "hi", "hello")
        await branch.execute(session, "set", str(start))

        await branch.execute(session, "fork")

        assert session.conv.uuid not in {None, start}
        assert set(db.get_branches()) == {start, session.conv.uuid}
        assert [m.content for m in session.conv.history.messages()] == ["hi", "hello"]

    async def test_without_a_started_conversation_is_reported(
        self, branch: Branch, session: Session, db: SqlAlchemy
    ) -> None:
        await branch.execute(session, "fork")

        assert _system(session) == ["You cannot fork without a conversation started/setted"]
        assert db.get_branches() == {}

    async def test_failed_creation_is_reported_and_nothing_is_loaded(
        self, branch: Branch, session: Session, db: SqlAlchemy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        start = _new_branch(db)
        await branch.execute(session, "set", str(start))
        _sent(session).clear()
        monkeypatch.setattr(db, "create_branch", lambda **_: None)

        await branch.execute(session, "fork")

        assert _system(session) == ["Failed to make a new branch"]
        assert session.conv.uuid == start


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("12345678-1234-5678-1234-567812345678", UUID("12345678-1234-5678-1234-567812345678")),
        ("not-a-uuid", None),
        ("", None),
    ],
)
def test_try_parse_uuid(value: str, expected: UUID | None) -> None:
    assert Branch._try_parse_uuid(value) == expected  # noqa: SLF001
