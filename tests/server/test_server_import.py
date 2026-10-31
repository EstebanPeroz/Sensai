import json
from pathlib import Path
from uuid import UUID

import pytest
from sensai_protocol import ClientCommand, ReadFileParams, System, WriteFileParams
from sensai_protocol.server_messages import Role
from sensai_server.commands._export import Export
from sensai_server.commands._import import Import
from sensai_server.llm.persona import PersonaManager
from sensai_server.llm.provider import ProviderManager
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai_server.session import Session, SessionRegistry
from sensai_server.tools.tool_manager import ToolManager

from tests.server.conftest import MODEL, SETTINGS, db_url
from tests.server.fakes import AnsweringOutbound

PATH = "~/saved.json"
CONFIG = {
    "branch": {"name": "Imported", "model": MODEL},
    "messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
}


@pytest.fixture
def db(tmp_path: Path) -> SqlAlchemy:
    return SqlAlchemy(db_url(tmp_path))


@pytest.fixture
def sessions(db: SqlAlchemy) -> SessionRegistry:
    return SessionRegistry(ProviderManager(SETTINGS.llm), PersonaManager(db), ToolManager())


async def _open(
    sessions: SessionRegistry, answers: dict[ClientCommand, dict | str]
) -> tuple[Session, AnsweringOutbound]:
    """A session whose client answers client commands with `answers`, a `ctx` or an `error` per command."""
    out = AnsweringOutbound(answers)
    session = await sessions.open(out, MODEL)
    assert session is not None
    out.sent.clear()
    return session, out


def _reading(content: str) -> dict[ClientCommand, dict | str]:
    """A client whose `read_file` answers with `content`."""
    return {ClientCommand.READ_FILE: {"content": content}}


def _system(out: AnsweringOutbound) -> str:
    message = out.sent[-1]
    assert isinstance(message, System)
    return message.message


def _messages(db: SqlAlchemy, branch: UUID) -> list[dict]:
    export = db.export_branch(branch)
    assert export is not None
    return export["messages"]


class TestImport:
    async def test_import_creates_the_branch(self, db: SqlAlchemy, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions, _reading(json.dumps(CONFIG)))
        before = set(db.get_branches())
        await Import(db).execute(session, PATH)
        (request,) = out.requests()
        assert ReadFileParams.model_validate(request.params).path == PATH
        (new,) = set(db.get_branches()) - before
        assert db.get_branches()[new] == "Imported"
        assert _messages(db, new) == CONFIG["messages"]
        assert _system(out) == f"Branch imported from {PATH}"

    async def test_client_error_is_reported(self, db: SqlAlchemy, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions, {ClientCommand.READ_FILE: "file not found"})
        before = db.get_branches()
        await Import(db).execute(session, PATH)
        assert db.get_branches() == before
        assert _system(out) == f"Cannot read {PATH}: file not found"

    async def test_invalid_json_is_reported(self, db: SqlAlchemy, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions, _reading("{not json"))
        before = db.get_branches()
        await Import(db).execute(session, PATH)
        assert db.get_branches() == before
        assert _system(out).startswith(f"Invalid JSON in {PATH}: ")

    async def test_json_list_is_reported(self, db: SqlAlchemy, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions, _reading("[1, 2]"))
        before = db.get_branches()
        await Import(db).execute(session, PATH)
        assert db.get_branches() == before
        assert _system(out) == f"Expected a JSON object in {PATH}, got list"

    async def test_malformed_config_is_reported(self, db: SqlAlchemy, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions, _reading(json.dumps({"messages": []})))
        before = db.get_branches()
        await Import(db).execute(session, PATH)
        assert db.get_branches() == before
        assert _system(out).startswith("[DatabaseError]")

    @pytest.mark.parametrize("args", [(), ("a.json", "b.json")])
    async def test_wrong_number_of_arguments(
        self, db: SqlAlchemy, sessions: SessionRegistry, args: tuple[str, ...]
    ) -> None:
        session, out = await _open(sessions, _reading(json.dumps(CONFIG)))
        await Import(db).execute(session, *args)
        assert out.requests() == []
        assert _system(out) == f"Invalid arguments. Usage: {Import.usage}"

    async def test_no_completions(self, db: SqlAlchemy, sessions: SessionRegistry) -> None:
        session, _ = await _open(sessions, _reading(""))
        assert Import(db).completions(session, []) == []

    async def test_export_then_import_keeps_the_messages(self, db: SqlAlchemy, sessions: SessionRegistry) -> None:
        branch = db.create_branch(model=MODEL, name="Saved")
        assert branch is not None
        db.add_message_to_branch(branch, Role.USER, "hello")
        db.add_message_to_branch(branch, Role.ASSISTANT, "hi")
        session, out = await _open(sessions, {ClientCommand.WRITE_FILE: {"path": "/home/user/saved.json"}})
        await Export(db).execute(session, str(branch))
        (request,) = out.requests()
        content = WriteFileParams.model_validate(request.params).content

        session, out = await _open(sessions, _reading(content))
        before = set(db.get_branches())
        await Import(db).execute(session, "saved.json")
        (new,) = set(db.get_branches()) - before
        assert db.get_branches()[new] == "Saved 2"
        assert _messages(db, new) == _messages(db, branch)
