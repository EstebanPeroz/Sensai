import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sensai_protocol import ClientCommand, System, WriteFileParams
from sensai_protocol.server_messages import Role
from sensai_server.commands._export import Export
from sensai_server.llm.persona import PersonaManager
from sensai_server.llm.provider import ProviderManager
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai_server.session import Session, SessionRegistry
from sensai_server.tools.tool_manager import ToolManager

from tests.server.conftest import MODEL, SETTINGS, db_url
from tests.server.fakes import AnsweringOutbound

WRITTEN = "/home/user/out.json"


@pytest.fixture
def db(tmp_path: Path) -> SqlAlchemy:
    return SqlAlchemy(db_url(tmp_path))


@pytest.fixture
def sessions(db: SqlAlchemy) -> SessionRegistry:
    return SessionRegistry(ProviderManager(SETTINGS.llm), PersonaManager(db), ToolManager())


@pytest.fixture
def branch(db: SqlAlchemy) -> UUID:
    """A branch named `Saved` holding one exchange."""
    branch = db.create_branch(model=MODEL, name="Saved")
    assert branch is not None
    db.add_message_to_branch(branch, Role.USER, "hello")
    db.add_message_to_branch(branch, Role.ASSISTANT, "hi")
    return branch


async def _open(sessions: SessionRegistry, answer: dict | str) -> tuple[Session, AnsweringOutbound]:
    """A session whose client answers `write_file` with `answer`, a `ctx` or an `error`."""
    out = AnsweringOutbound({ClientCommand.WRITE_FILE: answer})
    session = await sessions.open(out, MODEL)
    assert session is not None
    out.sent.clear()
    return session, out


def _system(out: AnsweringOutbound) -> str:
    message = out.sent[-1]
    assert isinstance(message, System)
    return message.message


class TestExport:
    async def test_export_current_asks_the_client_to_write_the_branch(
        self, db: SqlAlchemy, sessions: SessionRegistry, branch: UUID
    ) -> None:
        session, out = await _open(sessions, {"path": WRITTEN})
        session.conv.uuid = branch
        await Export(db).execute(session, "current")
        (request,) = out.requests()
        params = WriteFileParams.model_validate(request.params)
        assert params.path == f"{branch}.json"
        assert json.loads(params.content) == {
            "branch": {"name": "Saved", "model": MODEL},
            "messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        }
        assert _system(out) == f"Branch {branch} exported to {WRITTEN}"

    async def test_export_uuid_to_a_given_path(self, db: SqlAlchemy, sessions: SessionRegistry, branch: UUID) -> None:
        session, out = await _open(sessions, {"path": WRITTEN})
        await Export(db).execute(session, str(branch), "~/saved.json")
        (request,) = out.requests()
        assert WriteFileParams.model_validate(request.params).path == "~/saved.json"
        assert _system(out) == f"Branch {branch} exported to {WRITTEN}"

    async def test_no_conversation(self, db: SqlAlchemy, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions, {"path": WRITTEN})
        session.conv.uuid = None
        await Export(db).execute(session, "current")
        assert out.requests() == []
        assert _system(out) == "No conversation to export."

    async def test_invalid_uuid(self, db: SqlAlchemy, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions, {"path": WRITTEN})
        await Export(db).execute(session, "nope")
        assert out.requests() == []
        assert _system(out) == "Invalid uuid."

    async def test_unknown_branch(self, db: SqlAlchemy, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions, {"path": WRITTEN})
        unknown = uuid4()
        await Export(db).execute(session, str(unknown))
        assert out.requests() == []
        assert _system(out) == f"Branch {unknown} not found."

    async def test_client_error_is_reported(self, db: SqlAlchemy, sessions: SessionRegistry, branch: UUID) -> None:
        session, out = await _open(sessions, "directory not found")
        await Export(db).execute(session, str(branch))
        assert _system(out) == f"Failed to export branch {branch}: directory not found"

    @pytest.mark.parametrize("args", [(), ("current", "a.json", "extra")])
    async def test_wrong_number_of_arguments(
        self, db: SqlAlchemy, sessions: SessionRegistry, args: tuple[str, ...]
    ) -> None:
        session, out = await _open(sessions, {"path": WRITTEN})
        await Export(db).execute(session, *args)
        assert _system(out) == f"Usage: {Export.usage}"

    async def test_completions_list_current_and_the_branch_uuids(
        self, db: SqlAlchemy, sessions: SessionRegistry, branch: UUID
    ) -> None:
        session, _ = await _open(sessions, {"path": WRITTEN})
        export = Export(db)
        assert export.completions(session, []) == ["current", *map(str, db.get_branches())]
        assert str(branch) in export.completions(session, [])
        assert export.completions(session, ["current"]) == []
