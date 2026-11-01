from collections.abc import Callable
from typing import override

import pytest
from sensai_protocol import System
from sensai_server.commands._tools import ToolsCommand
from sensai_server.llm.persona import PersonaManager
from sensai_server.llm.provider import ProviderManager
from sensai_server.session import Session
from sensai_server.tools.state import ToolState
from sensai_server.tools.tool import Tool, ToolContext
from sensai_server.tools.tool_manager import ToolManager

from tests.server.conftest import SETTINGS
from tests.server.fakes import RecordingOutbound

USAGE = "Usage: /tools [list|enable|disable] [name]"

type Opener = Callable[[ToolManager], Session]


class StubTool(Tool[str]):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def description(self) -> str:
        return f"The {self._name} tool."

    @property
    @override
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    @override
    async def run(self, ctx: ToolContext, **kwargs: object) -> str:
        del ctx, kwargs
        return f"{self._name} ran"


def make_manager(disabled: tuple[str, ...] = ()) -> ToolManager:
    return ToolManager([StubTool("weather"), StubTool("echo")], disabled=disabled)


def enabled_names(manager: ToolManager, state: ToolState) -> list[str]:
    return [name for name in manager.names() if manager.is_enabled(state, name)]


def sent(session: Session) -> list[tuple[str, bool]]:
    """Every system message sent to the client of `session`, with whether it joins the previous response."""
    out = session.out
    assert isinstance(out, RecordingOutbound)
    return [(message.message, message.append) for message in out.sent if isinstance(message, System)]


def messages(session: Session) -> list[str]:
    return [message for message, _ in sent(session)]


@pytest.fixture
def open_session(persona_manager: PersonaManager) -> Opener:
    """Open a session over `tool_manager`, as `SessionRegistry.open` does, recording what it is sent."""
    pm = ProviderManager(SETTINGS.llm)

    def open_(tool_manager: ToolManager) -> Session:
        return Session(RecordingOutbound(), pm, persona_manager, tool_manager)

    return open_


@pytest.fixture
def tool_manager() -> ToolManager:
    return make_manager()


@pytest.fixture
def session(open_session: Opener, tool_manager: ToolManager) -> Session:
    return open_session(tool_manager)


class TestToolsList:
    async def test_no_arguments_lists_every_tool_with_its_state(
        self, session: Session, tool_manager: ToolManager
    ) -> None:
        await ToolsCommand(tool_manager).execute(session)

        assert messages(session) == ["Available tools:\n- echo: enabled\n- weather: enabled"]

    async def test_list_argument_lists_every_tool_with_its_state(
        self, session: Session, tool_manager: ToolManager
    ) -> None:
        await ToolsCommand(tool_manager).execute(session, "list")

        assert messages(session) == ["Available tools:\n- echo: enabled\n- weather: enabled"]

    async def test_tools_are_listed_in_alphabetical_order(self, open_session: Opener) -> None:
        manager = ToolManager([StubTool("zebra"), StubTool("alpha"), StubTool("middle")])
        session = open_session(manager)

        await ToolsCommand(manager).execute(session, "list")

        lines = [line for line in messages(session)[0].splitlines() if line.startswith("- ")]
        assert lines == ["- alpha: enabled", "- middle: enabled", "- zebra: enabled"]

    async def test_disabled_tools_are_reported_as_disabled(self, open_session: Opener) -> None:
        manager = make_manager(disabled=("echo",))
        session = open_session(manager)

        await ToolsCommand(manager).execute(session, "list")

        assert messages(session) == ["Available tools:\n- echo: disabled\n- weather: enabled"]

    async def test_tools_asking_confirmation_are_reported(self, open_session: Opener) -> None:
        manager = ToolManager([StubTool("weather"), StubTool("echo")], confirm=("echo",))
        session = open_session(manager)

        await ToolsCommand(manager).execute(session, "list")

        assert messages(session) == ["Available tools:\n- echo: enabled, asks confirmation\n- weather: enabled"]

    async def test_always_allowed_tool_no_longer_asks_confirmation(self, open_session: Opener) -> None:
        manager = ToolManager([StubTool("echo")], confirm=("echo",))
        session = open_session(manager)
        session.tools.always_allowed.add("echo")

        await ToolsCommand(manager).execute(session, "list")

        assert messages(session) == ["Available tools:\n- echo: enabled"]

    async def test_listing_without_any_registered_tool(self, open_session: Opener) -> None:
        manager = ToolManager()
        session = open_session(manager)

        await ToolsCommand(manager).execute(session, "list")

        assert messages(session) == ["No tools are available."]

    async def test_extra_arguments_after_list_are_ignored(self, session: Session, tool_manager: ToolManager) -> None:
        await ToolsCommand(tool_manager).execute(session, "list", "unused", "args")

        assert messages(session) == ["Available tools:\n- echo: enabled\n- weather: enabled"]

    async def test_listing_leaves_every_tool_state_untouched(self, open_session: Opener) -> None:
        manager = make_manager(disabled=("echo",))
        session = open_session(manager)

        await ToolsCommand(manager).execute(session, "list")

        assert enabled_names(manager, session.tools) == ["weather"]


class TestToolsEnable:
    async def test_enables_a_disabled_tool(self, open_session: Opener) -> None:
        manager = make_manager(disabled=("echo",))
        session = open_session(manager)

        await ToolsCommand(manager).execute(session, "enable", "echo")

        assert manager.is_enabled(session.tools, "echo")
        assert messages(session) == ["Tool 'echo' enabled."]

    async def test_enabling_an_already_enabled_tool_keeps_it_enabled(
        self, session: Session, tool_manager: ToolManager
    ) -> None:
        await ToolsCommand(tool_manager).execute(session, "enable", "echo")

        assert enabled_names(tool_manager, session.tools) == ["echo", "weather"]
        assert messages(session) == ["Tool 'echo' enabled."]

    async def test_enabling_only_affects_the_named_tool(self, open_session: Opener) -> None:
        manager = make_manager(disabled=("echo", "weather"))
        session = open_session(manager)

        await ToolsCommand(manager).execute(session, "enable", "echo")

        assert enabled_names(manager, session.tools) == ["echo"]

    async def test_unknown_tool_name_is_reported_and_changes_nothing(
        self, session: Session, tool_manager: ToolManager
    ) -> None:
        await ToolsCommand(tool_manager).execute(session, "enable", "ghost")

        assert enabled_names(tool_manager, session.tools) == ["echo", "weather"]
        assert messages(session) == ["Tool 'ghost' is not available."]

    async def test_missing_tool_name_displays_usage(self, session: Session, tool_manager: ToolManager) -> None:
        await ToolsCommand(tool_manager).execute(session, "enable")

        assert messages(session) == [USAGE]

    async def test_too_many_arguments_displays_usage(self, open_session: Opener) -> None:
        manager = make_manager(disabled=("echo",))
        session = open_session(manager)

        await ToolsCommand(manager).execute(session, "enable", "echo", "weather")

        assert not manager.is_enabled(session.tools, "echo")
        assert messages(session) == [USAGE]


class TestToolsDisable:
    async def test_disables_an_enabled_tool(self, session: Session, tool_manager: ToolManager) -> None:
        await ToolsCommand(tool_manager).execute(session, "disable", "echo")

        assert not tool_manager.is_enabled(session.tools, "echo")
        assert messages(session) == ["Tool 'echo' disabled."]

    async def test_disabling_an_already_disabled_tool_keeps_it_disabled(self, open_session: Opener) -> None:
        manager = make_manager(disabled=("echo",))
        session = open_session(manager)

        await ToolsCommand(manager).execute(session, "disable", "echo")

        assert enabled_names(manager, session.tools) == ["weather"]
        assert messages(session) == ["Tool 'echo' disabled."]

    async def test_disabling_only_affects_the_named_tool(self, session: Session, tool_manager: ToolManager) -> None:
        await ToolsCommand(tool_manager).execute(session, "disable", "echo")

        assert enabled_names(tool_manager, session.tools) == ["weather"]

    async def test_unknown_tool_name_is_reported_and_changes_nothing(
        self, session: Session, tool_manager: ToolManager
    ) -> None:
        await ToolsCommand(tool_manager).execute(session, "disable", "ghost")

        assert enabled_names(tool_manager, session.tools) == ["echo", "weather"]
        assert messages(session) == ["Tool 'ghost' is not available."]

    async def test_missing_tool_name_displays_usage(self, session: Session, tool_manager: ToolManager) -> None:
        await ToolsCommand(tool_manager).execute(session, "disable")

        assert enabled_names(tool_manager, session.tools) == ["echo", "weather"]
        assert messages(session) == [USAGE]

    async def test_too_many_arguments_displays_usage(self, session: Session, tool_manager: ToolManager) -> None:
        await ToolsCommand(tool_manager).execute(session, "disable", "echo", "weather")

        assert enabled_names(tool_manager, session.tools) == ["echo", "weather"]
        assert messages(session) == [USAGE]


class TestToolsInvalidAction:
    async def test_unknown_action_displays_usage(self, session: Session, tool_manager: ToolManager) -> None:
        await ToolsCommand(tool_manager).execute(session, "bogus")

        assert messages(session) == [USAGE]

    async def test_unknown_action_with_a_tool_name_displays_usage(
        self, session: Session, tool_manager: ToolManager
    ) -> None:
        await ToolsCommand(tool_manager).execute(session, "toggle", "echo")

        assert enabled_names(tool_manager, session.tools) == ["echo", "weather"]
        assert messages(session) == [USAGE]

    @pytest.mark.parametrize("action", ["LIST", "Enable", "DISABLE"])
    async def test_actions_are_case_sensitive(self, session: Session, tool_manager: ToolManager, action: str) -> None:
        await ToolsCommand(tool_manager).execute(session, action, "echo")

        assert messages(session) == [USAGE]


class TestToolsCommandMetadata:
    def test_registry_metadata_is_exposed(self) -> None:
        assert ToolsCommand.name == "tools"
        assert ToolsCommand.help == "List available tools and toggle them."
        assert ToolsCommand.usage == "/tools [list|enable|disable] [name]"


class TestToolsCompletions:
    def test_no_argument_completes_the_subcommand_names(self, session: Session, tool_manager: ToolManager) -> None:
        assert ToolsCommand(tool_manager).completions(session, []) == ["disable", "enable", "list"]

    def test_subcommand_names_are_completed_even_without_any_tool(self, session: Session) -> None:
        assert ToolsCommand(ToolManager()).completions(session, []) == ["disable", "enable", "list"]

    @pytest.mark.parametrize("subcommand", ["enable", "disable"])
    def test_toggling_subcommands_complete_the_tool_names(self, session: Session, subcommand: str) -> None:
        manager = ToolManager([StubTool("zebra"), StubTool("alpha"), StubTool("middle")])

        assert ToolsCommand(manager).completions(session, [subcommand]) == ["alpha", "middle", "zebra"]

    def test_disabled_tools_are_completed_too(self, open_session: Opener) -> None:
        # `/tools enable <name>` must be able to complete a name that is currently disabled.
        manager = make_manager(disabled=("echo",))

        assert ToolsCommand(manager).completions(open_session(manager), ["enable"]) == ["echo", "weather"]

    def test_no_tool_name_is_completed_without_any_registered_tool(self, session: Session) -> None:
        assert ToolsCommand(ToolManager()).completions(session, ["enable"]) == []

    def test_list_subcommand_takes_no_argument(self, session: Session, tool_manager: ToolManager) -> None:
        assert ToolsCommand(tool_manager).completions(session, ["list"]) == []

    def test_unknown_subcommand_completes_nothing(self, session: Session, tool_manager: ToolManager) -> None:
        assert ToolsCommand(tool_manager).completions(session, ["bogus"]) == []

    @pytest.mark.parametrize("args", [["enable", "echo"], ["disable", "echo"], ["list", "echo"]])
    def test_nothing_is_completed_past_the_tool_name(
        self, session: Session, tool_manager: ToolManager, args: list[str]
    ) -> None:
        assert ToolsCommand(tool_manager).completions(session, args) == []

    def test_completing_sends_nothing_to_the_client(self, session: Session, tool_manager: ToolManager) -> None:
        ToolsCommand(tool_manager).completions(session, [])

        assert sent(session) == []

    def test_completing_leaves_every_tool_state_untouched(self, open_session: Opener) -> None:
        manager = make_manager(disabled=("echo",))
        session = open_session(manager)

        ToolsCommand(manager).completions(session, ["enable"])

        assert enabled_names(manager, session.tools) == ["weather"]


class TestToolsCompletionMatchesExecute:
    @pytest.mark.parametrize("subcommand", ["list", "enable", "disable"])
    async def test_every_completed_subcommand_is_accepted_by_execute(
        self, session: Session, tool_manager: ToolManager, subcommand: str
    ) -> None:
        command = ToolsCommand(tool_manager)

        assert subcommand in command.completions(session, [])
        await command.execute(session, subcommand, "echo")

        assert messages(session) != [USAGE]

    @pytest.mark.parametrize("subcommand", ["set", "remove"])
    async def test_unknown_subcommand_names_are_rejected(
        self, session: Session, tool_manager: ToolManager, subcommand: str
    ) -> None:
        await ToolsCommand(tool_manager).execute(session, subcommand, "echo")

        assert enabled_names(tool_manager, session.tools) == ["echo", "weather"]
        assert messages(session) == [USAGE]


class TestToolsEffectOnModelPayload:
    async def test_disabling_a_tool_removes_its_rule_from_the_payload(
        self, session: Session, tool_manager: ToolManager
    ) -> None:
        await ToolsCommand(tool_manager).execute(session, "disable", "echo")

        assert tool_manager.get_tool_call_rules(session.tools) == [StubTool("weather").tool_rule]

    async def test_enabling_a_tool_restores_its_rule_in_the_payload(self, open_session: Opener) -> None:
        manager = make_manager(disabled=("echo", "weather"))
        session = open_session(manager)

        await ToolsCommand(manager).execute(session, "enable", "weather")

        assert manager.get_tool_call_rules(session.tools) == [StubTool("weather").tool_rule]

    async def test_disabled_tool_becomes_unreachable_for_the_model(
        self, session: Session, tool_manager: ToolManager
    ) -> None:
        call = {"type": "function", "function": {"name": "echo", "arguments": {}}}

        await ToolsCommand(tool_manager).execute(session, "disable", "echo")

        result = await tool_manager.call(call, session.tools, ToolContext(out=session.out))
        assert result == {"role": "tool", "tool_name": "echo", "content": "error: unknown tool 'echo'"}


class TestToolsResponseAppending:
    async def test_enable_message_is_appended_to_the_response(self, open_session: Opener) -> None:
        manager = make_manager(disabled=("echo",))
        session = open_session(manager)

        await ToolsCommand(manager).execute(session, "enable", "echo")

        assert sent(session) == [("Tool 'echo' enabled.", True)]

    async def test_disable_message_is_appended_to_the_response(
        self, session: Session, tool_manager: ToolManager
    ) -> None:
        await ToolsCommand(tool_manager).execute(session, "disable", "echo")

        assert sent(session) == [("Tool 'echo' disabled.", True)]

    async def test_listing_is_not_appended_to_the_response(self, session: Session, tool_manager: ToolManager) -> None:
        await ToolsCommand(tool_manager).execute(session, "list")

        assert sent(session) == [("Available tools:\n- echo: enabled\n- weather: enabled", False)]

    @pytest.mark.parametrize("subcommand", ["enable", "disable"])
    async def test_unknown_tool_message_is_not_appended_to_the_response(
        self, session: Session, tool_manager: ToolManager, subcommand: str
    ) -> None:
        await ToolsCommand(tool_manager).execute(session, subcommand, "ghost")

        assert sent(session) == [("Tool 'ghost' is not available.", False)]

    async def test_usage_message_is_not_appended_to_the_response(
        self, session: Session, tool_manager: ToolManager
    ) -> None:
        await ToolsCommand(tool_manager).execute(session, "enable")

        assert sent(session) == [(USAGE, False)]


class TestToolsPerSession:
    async def test_two_sessions_do_not_share_their_state(self, open_session: Opener, tool_manager: ToolManager) -> None:
        first, second = open_session(tool_manager), open_session(tool_manager)
        command = ToolsCommand(tool_manager)

        await command.execute(first, "disable", "echo")

        assert enabled_names(tool_manager, first.tools) == ["weather"]
        assert enabled_names(tool_manager, second.tools) == ["echo", "weather"]

    async def test_each_session_lists_its_own_state(self, open_session: Opener, tool_manager: ToolManager) -> None:
        first, second = open_session(tool_manager), open_session(tool_manager)
        command = ToolsCommand(tool_manager)
        await command.execute(first, "disable", "echo")

        await command.execute(first, "list")
        await command.execute(second, "list")

        assert messages(first)[-1] == "Available tools:\n- echo: disabled\n- weather: enabled"
        assert messages(second) == ["Available tools:\n- echo: enabled\n- weather: enabled"]

    async def test_replies_go_to_the_calling_session_only(
        self, open_session: Opener, tool_manager: ToolManager
    ) -> None:
        first, second = open_session(tool_manager), open_session(tool_manager)

        await ToolsCommand(tool_manager).execute(first, "disable", "echo")

        assert messages(first) == ["Tool 'echo' disabled."]
        assert sent(second) == []

    async def test_a_new_session_starts_from_the_settings_not_from_the_others(self, open_session: Opener) -> None:
        manager = make_manager(disabled=("echo",))
        first = open_session(manager)
        await ToolsCommand(manager).execute(first, "enable", "echo")

        later = open_session(manager)

        assert enabled_names(manager, later.tools) == ["weather"]
