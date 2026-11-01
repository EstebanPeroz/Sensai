from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sensai_protocol import Dashboard as DashboardMessage
from sensai_protocol import Effort as EffortMessage
from sensai_protocol import Period, System
from sensai_server.chat.react import EFFORTS
from sensai_server.commands import CommandRegistry, PersonaCommand, build_commands
from sensai_server.commands._base import Command
from sensai_server.commands._branch import Branch
from sensai_server.commands._dashboard import Dashboard
from sensai_server.commands._effort import Effort
from sensai_server.commands._export import Export
from sensai_server.commands._help import Help
from sensai_server.commands._import import Import
from sensai_server.commands._model import Model
from sensai_server.commands._tools import ToolsCommand
from sensai_server.commands.registry import UnknownCommandError
from sensai_server.llm.persona import Persona, PersonaManager
from sensai_server.llm.provider import ProviderManager
from sensai_server.memory import Conversation
from sensai_server.memory.persistent_db.error import DatabaseError
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai_server.monitoring.interaction import Interaction, InteractionStatus
from sensai_server.monitoring.interaction_repository import InteractionRepository
from sensai_server.monitoring.metrics import history_start
from sensai_server.session import Session, SessionRegistry
from sensai_server.tools.tool_manager import ToolManager

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
def pm() -> ProviderManager:
    return ProviderManager(SETTINGS.llm)


@pytest.fixture
def tool_manager() -> ToolManager:
    return ToolManager()


@pytest.fixture
def sessions(pm: ProviderManager, manager: PersonaManager, tool_manager: ToolManager) -> SessionRegistry:
    return SessionRegistry(pm, manager, tool_manager)


@pytest.fixture
def commands(
    sessions: SessionRegistry, db: SqlAlchemy, pm: ProviderManager, tool_manager: ToolManager
) -> CommandRegistry:
    """The server's commands, built as `create_app` does."""
    commands = CommandRegistry()
    commands.set_commands(build_commands(sessions, commands, db, pm, tool_manager))
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


class TestCommand:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            Command()  # type: ignore[abstract]

    def test_subclass_without_execute_cannot_be_instantiated(self) -> None:
        class Incomplete(Command):
            name = "incomplete"
            help = "Missing execute."

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    async def test_has_no_completions_by_default(self, sessions: SessionRegistry) -> None:
        class Plain(Command):
            name = "plain"
            help = "Does nothing."

            async def execute(self, session: Session, *args: str) -> None:
                del session, args

        session, _ = await _open(sessions)
        assert Plain().completions(session, ["a"]) == []


class TestUnknownCommandError:
    def test_stores_command_name_and_message(self) -> None:
        error = UnknownCommandError("nope")

        assert error.command_name == "nope"
        assert str(error) == "Unknown command: nope"


class TestCommandRegistry:
    async def test_names(self, commands: CommandRegistry) -> None:
        assert commands.names() == [
            "/model",
            "/persona",
            "/branch",
            "/tools",
            "/effort",
            "/export",
            "/import",
            "/dashboard",
            "/help",
        ]

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

    def test_commands_are_the_registered_ones(self) -> None:
        echo = _Echo()
        assert _registry(echo).commands() == [echo]

    def test_a_later_command_replaces_one_of_the_same_name(self) -> None:
        first, second = _Echo(), _Echo()
        assert _registry(first, second).commands() == [second]

    async def test_an_empty_registry_reports_every_command_as_unknown(self, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        await _registry().execute(session, "/echo x")
        assert _system(out) == "Unknown command: echo"

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
        assert _system(out) == (
            f"Available commands:\n- branch: {Branch.help}\n- dashboard: {Dashboard.help}\n- effort: {Effort.help}"
            f"\n- export: {Export.help}\n- help: {Help.help}\n- import: {Import.help}\n- model: {Model.help}"
            f"\n- persona: {PersonaCommand.help}\n- tools: {ToolsCommand.help}"
        )

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
        branch = db.create_branch(model=MODEL)
        assert branch is not None
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


class TestEffortCommand:
    async def test_display(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        await commands.execute(session, "/effort")
        assert _system(out).startswith(f"Current effort: {session.effort}\nAvailable efforts:\n- off:")

    async def test_display_describes_every_level(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        await commands.execute(session, "/effort")
        assert _system(out) == (
            f"Current effort: {session.effort}\n"
            "Available efforts:\n"
            "- off: answer directly, without thinking nor tools\n"
            f"- low: up to {EFFORTS['low']} thinking steps\n"
            f"- medium: up to {EFFORTS['medium']} thinking steps\n"
            f"- high: up to {EFFORTS['high']} thinking steps"
        )

    @pytest.mark.parametrize("level", list(EFFORTS))
    async def test_every_level_can_be_set_and_is_shown_to_the_client(
        self, commands: CommandRegistry, sessions: SessionRegistry, level: str
    ) -> None:
        session, out = await _open(sessions)
        await commands.execute(session, f"/effort {level}")
        assert session.effort == level
        assert out.sent[0] == EffortMessage(level=level)
        assert _system(out).startswith(f"Effort set to '{level}': ")

    async def test_set_only_changes_its_session(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        first, _ = await _open(sessions)
        second, _ = await _open(sessions)
        before = second.effort
        await commands.execute(first, "/effort off")
        assert second.effort == before

    async def test_set(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        await commands.execute(session, "/effort low")
        assert session.effort == "low"
        assert out.sent[0] == EffortMessage(level="low")
        assert _system(out) == "Effort set to 'low': up to 1 thinking steps."

    async def test_unknown_effort_is_reported(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        before = session.effort
        await commands.execute(session, "/effort extreme")
        assert session.effort == before
        assert not any(isinstance(message, EffortMessage) for message in out.sent)
        assert _system(out) == f"Effort 'extreme' does not exist. Usage: {Effort.usage}"

    async def test_too_many_arguments(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        before = session.effort
        await commands.execute(session, "/effort low high")
        assert session.effort == before
        assert _system(out) == f"Invalid number of arguments. Usage: {Effort.usage}"

    async def test_completions(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        session, _ = await _open(sessions)
        assert commands.completions(session, "/effort ") == list(EFFORTS)


class TestModelCommand:
    async def test_display(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        await commands.execute(session, "/model")
        assert _system(out) == f"Current model: {MODEL}\nAvailable models:\n- a\n- b"

    async def test_display_without_models(
        self, commands: CommandRegistry, sessions: SessionRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session, out = await _open(sessions)
        monkeypatch.setattr(session.provider_manager, "get_models", list)
        await commands.execute(session, "/model")
        assert _system(out) == "No models are available."

    async def test_set(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        await commands.execute(session, "/model b")
        assert session.conv.model == "b"
        assert _system(out) == "Model set to 'b'."

    async def test_unknown_model_is_reported(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        await commands.execute(session, "/model nope")
        assert session.conv.model == MODEL
        assert _system(out) == "Model 'nope' is not available."

    async def test_too_many_arguments(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        await commands.execute(session, "/model b a")
        assert session.conv.model == MODEL
        assert _system(out) == f"Invalid number of arguments. Usage: {Model.usage}"

    async def test_completions(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        session, _ = await _open(sessions)
        assert commands.completions(session, "/model ") == ["a", "b"]


NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


def _dashboard_registry(repository: InteractionRepository) -> CommandRegistry:
    return _registry(Dashboard(repository, clock=lambda: NOW))


def _report(out: RecordingOutbound) -> DashboardMessage:
    """The last message sent, which must be a `dashboard` one."""
    message = out.sent[-1]
    assert isinstance(message, DashboardMessage)
    return message


class TestDashboardCommand:
    @pytest.mark.parametrize(
        ("text", "period"),
        [
            ("/dashboard", Period.DAY),
            ("/dashboard 24h", Period.DAY),
            ("/dashboard 7d", Period.WEEK),
            ("/dashboard 30d", Period.MONTH),
        ],
    )
    async def test_sends_the_report_of_the_period(self, sessions: SessionRegistry, text: str, period: Period) -> None:
        session, out = await _open(sessions)
        repository = MagicMock(spec=InteractionRepository)
        repository.get_all.return_value = []
        await _dashboard_registry(repository).execute(session, text)
        repository.get_all.assert_called_once_with(since=history_start(period, NOW), with_content=False)
        report = _report(out).report
        assert report.period == period
        assert report.end == NOW

    async def test_the_report_counts_the_logged_calls(self, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        repository = MagicMock(spec=InteractionRepository)
        repository.get_all.return_value = [
            Interaction(
                model="qwen", status=InteractionStatus.OK, latency_ms=1000.0, created_at=NOW - timedelta(hours=1)
            ),
            Interaction(
                model="qwen",
                status=InteractionStatus.ERROR,
                error="boom",
                latency_ms=10.0,
                created_at=NOW - timedelta(hours=2),
            ),
        ]
        await _dashboard_registry(repository).execute(session, "/dashboard")
        report = _report(out).report
        assert report.summary.calls == 2
        assert report.summary.errors == 1
        assert [error.error for error in report.recent_errors] == ["boom"]

    @pytest.mark.parametrize("text", ["/dashboard 1y", "/dashboard 7d x"])
    async def test_bad_arguments_answer_with_the_usage(self, sessions: SessionRegistry, text: str) -> None:
        session, out = await _open(sessions)
        repository = MagicMock(spec=InteractionRepository)
        await _dashboard_registry(repository).execute(session, text)
        assert _system(out).endswith(f"Usage: {Dashboard.usage}")
        repository.get_all.assert_not_called()

    async def test_a_database_error_is_reported(self, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        repository = MagicMock(spec=InteractionRepository)
        repository.get_all.side_effect = DatabaseError("unreadable")
        await _dashboard_registry(repository).execute(session, "/dashboard")
        assert _system(out) == "[DatabaseError]unreadable"

    async def test_reads_the_server_database(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        await commands.execute(session, "/dashboard 7d")
        assert _report(out).report.summary.calls == 0

    async def test_completions(self, commands: CommandRegistry, sessions: SessionRegistry) -> None:
        session, _ = await _open(sessions)
        assert commands.completions(session, "/dashboard ") == ["24h", "7d", "30d"]

    async def test_uses_the_current_time_by_default(self, sessions: SessionRegistry) -> None:
        session, out = await _open(sessions)
        repository = MagicMock(spec=InteractionRepository)
        repository.get_all.return_value = []
        before = datetime.now(tz=UTC)
        await _registry(Dashboard(repository)).execute(session, "/dashboard")
        assert before <= _report(out).report.end <= datetime.now(tz=UTC)
