from pathlib import Path
from uuid import uuid4

import pytest
from sensai_protocol import System
from sensai_server.commands import CommandRegistry, PersonaCommand, build_commands
from sensai_server.commands._base import Command
from sensai_server.commands._help import Help
from sensai_server.llm.persona import Persona, PersonaManager
from sensai_server.llm.provider import ProviderManager
from sensai_server.memory import Conversation
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai_server.session import Session, SessionRegistry

from tests.server.conftest import MODEL, SETTINGS, db_url
from tests.server.fakes import RecordingOutbound

ANALYST = "Analyst"


@pytest.fixture
def db(tmp_path: Path) -> SqlAlchemy:
    return SqlAlchemy(db_url(tmp_path))


@pytest.fixture
def manager(db: SqlAlchemy) -> PersonaManager:
    manager = PersonaManager(db)
    manager.create_default_persona()
    return manager


@pytest.fixture
def sessions(manager: PersonaManager) -> SessionRegistry:
    return SessionRegistry(ProviderManager(SETTINGS.llm), manager)


@pytest.fixture
def commands(sessions: SessionRegistry) -> CommandRegistry:
    """The server's commands, built as `create_app` does."""
    commands = CommandRegistry()
    commands.set_commands(build_commands(sessions, commands))
    return commands


def _registry(*commands: Command) -> CommandRegistry:
    registry = CommandRegistry()
    registry.set_commands(list(commands))
    return registry


async def _open(sessions: SessionRegistry) -> tuple[Session, RecordingOutbound]:
    out = RecordingOutbound()
    session = await sessions.open(out, MODEL)
    assert session is not None
    out.sent.clear()
    return session, out


def _system(out: RecordingOutbound) -> str:
    """The text of the last message sent, which must be a `system` one."""
    message = out.sent[-1]
    assert isinstance(message, System)
    return message.message


class _Echo(Command):
    name = "echo"
    help = "Echo the arguments."
    usage = "/echo <words>"

    def __init__(self) -> None:
        self.completed: list[list[str]] = []

    async def execute(self, session: Session, *args: str) -> None:
        await session.out.send_system_message(" ".join(args))

    def completions(self, session: Session, args: list[str]) -> list[str]:
        del session
        self.completed.append(args)
        return ["x"]


class TestCommandRegistry:
    async def test_names(self, commands: CommandRegistry) -> None:
        assert commands.names() == ["/persona", "/help"]

    async def test_execute_splits_the_arguments(self, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        await _registry(_Echo()).execute(session, "/echo  a   b")
        assert _system(out) == "a b"

    async def test_unknown_command_is_reported(self, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        await _registry(_Echo()).execute(session, "/nope x")
        assert _system(out) == "Unknown command: nope"

    async def test_empty_line_does_nothing(self, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        await _registry(_Echo()).execute(session, "/")
        assert out.sent == []

    @pytest.mark.parametrize(
        ("text", "finished"),
        [("/echo ", []), ("/echo a", []), ("/echo a ", ["a"]), ("/echo a b", ["a"])],
    )
    async def test_completions_get_the_finished_words(
        self, sessions: SessionRegistry, text: str, finished: list[str]
    ) -> None:
        session, _ = await _open(sessions)
        echo = _Echo()
        assert _registry(echo).completions(session, text) == ["x"]
        assert echo.completed == [finished]

    @pytest.mark.parametrize("text", ["", "hello", "/echo", "/nope "])
    async def test_no_completions(self, sessions: SessionRegistry, text: str) -> None:
        session, _ = await _open(sessions)
        echo = _Echo()
        assert _registry(echo).completions(session, text) == []
        assert echo.completed == []


class TestHelp:
    async def test_lists_every_command_by_name(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        await commands.execute(session, "/help")
        assert _system(out) == (f"Available commands:\n- help: {Help.help}\n- persona: {PersonaCommand.help}")

    async def test_lists_the_commands_of_its_registry(self, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        registry = CommandRegistry()
        registry.set_commands([_Echo(), Help(registry)])
        await registry.execute(session, "/help extra")
        assert _system(out) == f"Available commands:\n- echo: {_Echo.help}\n- help: {Help.help}"


class TestPersonaCommand:
    @pytest.mark.parametrize("line", ["/persona", "/persona nope", "/persona info", "/persona list x"])
    async def test_usage(self, commands: CommandRegistry, sessions: SessionRegistry, line: str) -> None:
        session, out = await _open(sessions)
        await commands.execute(session, line)
        assert _system(out).startswith("Usage: /persona list")

    async def test_list_marks_the_loaded_persona(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        await commands.execute(session, "/persona list")
        assert f"- {ANALYST}:" in _system(out)
        await commands.execute(session, f"/persona load {ANALYST}")
        await commands.execute(session, "/persona list")
        assert f"- {ANALYST} (loaded):" in _system(out)

    async def test_info(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        await commands.execute(session, "/persona info analyst")
        text = _system(out)
        assert f"Name: {ANALYST}" in text
        assert "Loaded: no" in text
        assert "Prompt:" in text

    async def test_unknown_persona_is_reported(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        await commands.execute(session, "/persona load nope")
        assert "persona 'nope' is not registered" in _system(out)
        assert session.conv.persona is None

    async def test_load_in_one_session_only(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        first, out = await _open(sessions)
        second, _ = await _open(sessions)
        await commands.execute(first, "/persona load analyst")
        assert _system(out) == f"Persona '{ANALYST}' loaded."
        assert first.conv.persona is not None
        assert first.conv.persona.name == ANALYST
        assert second.conv.persona is None

    async def test_unload(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        await commands.execute(session, "/persona unload")
        assert _system(out) == "No persona loaded."
        await commands.execute(session, f"/persona load {ANALYST}")
        await commands.execute(session, "/persona unload")
        assert _system(out) == f"Persona '{ANALYST}' unloaded."
        assert session.conv.persona is None

    async def test_load_saves_the_persona_on_the_branch(
        self, commands: CommandRegistry, sessions: SessionRegistry, db: SqlAlchemy
    ) -> None:
        session, out = await _open(sessions)
        branch = db.create_branch()
        assert branch is not None
        db.set_branch_model(branch, MODEL)
        session.conv.uuid = branch
        await commands.execute(session, f"/persona load {ANALYST}")
        assert _system(out) == f"Persona '{ANALYST}' loaded."
        restored = Conversation()
        db.get_branch(restored, branch, ProviderManager(SETTINGS.llm))
        assert restored.persona == session.conv.persona

    async def test_load_on_a_missing_branch_is_reported(
        self, commands: CommandRegistry, sessions: SessionRegistry
    ) -> None:
        session, out = await _open(sessions)
        session.conv.uuid = uuid4()
        await commands.execute(session, f"/persona load {ANALYST}")
        assert _system(out) == f"Persona '{ANALYST}' loaded, but not saved on the branch."
        assert session.conv.persona is not None

    async def test_update_refreshes_every_session(
        self, commands: CommandRegistry, sessions: SessionRegistry, manager: PersonaManager
    ) -> None:
        first, out = await _open(sessions)
        second, _ = await _open(sessions)
        await commands.execute(second, f"/persona load {ANALYST}")
        await commands.execute(first, f"/persona update {ANALYST} prompt=Answer in French. description=French")
        assert _system(out) == f"Persona '{ANALYST}' updated: prompt, description."
        assert second.conv.persona is not None
        assert second.conv.persona.prompt == "Answer in French."
        assert second.conv.persona.description == "French"
        assert manager.get_persona(ANALYST).prompt == "Answer in French."

    @pytest.mark.parametrize(
        ("assignments", "reason"),
        [("oops", "expected 'field=value'"), ("uuid=x", "cannot set 'uuid'"), ("name=a name=b", "set twice")],
    )
    async def test_update_refuses_bad_assignments(
        self,
        commands: CommandRegistry,
        sessions: SessionRegistry,
        manager: PersonaManager,
        assignments: str,
        reason: str,
    ) -> None:
        session, out = await _open(sessions)
        before: Persona = manager.get_persona(ANALYST)
        await commands.execute(session, f"/persona update {ANALYST} {assignments}")
        assert reason in _system(out)
        assert manager.get_persona(ANALYST) == before

    @pytest.mark.parametrize(
        ("text", "values"),
        [
            ("/persona ", ["create", "info", "list", "load", "unload", "update", "updateFromFile"]),
            ("/persona load ", [ANALYST]),
            ("/persona update ", [ANALYST]),
            ("/persona update Analyst ", ["description=", "name=", "prompt="]),
            ("/persona create ", []),
            ("/persona list ", []),
        ],
    )
    async def test_completions(
        self, commands: CommandRegistry, sessions: SessionRegistry, text: str, values: list[str]
    ) -> None:
        session, _ = await _open(sessions)
        assert commands.completions(session, text) == values
