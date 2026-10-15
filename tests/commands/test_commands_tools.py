from typing import override

import pytest

from sensai.commands.tools import ToolsCommand
from sensai.tools.tool import Tool
from sensai.tools.tool_manager import ToolManager
from tests.commands.fakes import FakeUI

USAGE = "Usage: /tools [list|enable|disable] [name]"


class StubTool(Tool[str]):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def tool_rule(self) -> dict:
        return {"type": "function", "function": {"name": self._name}}

    @override
    def run(self, **kwargs: object) -> str:
        del kwargs
        return f"{self._name} ran"


class RecordingUI(FakeUI):
    # FakeUI drops the `append_response` flag; record it to assert which messages join the response.
    def __init__(self) -> None:
        super().__init__()
        self.appended: list[tuple[str, bool]] = []

    @override
    def send_system_message(self, message: str, *, append_response: bool = False) -> bool:
        self.appended.append((message, append_response))
        return super().send_system_message(message, append_response=append_response)


def make_manager(disabled: tuple[str, ...] = ()) -> ToolManager:
    return ToolManager([StubTool("weather"), StubTool("echo")], disabled=disabled)


def enabled_names(manager: ToolManager) -> list[str]:
    return [name for name in manager.names() if manager.is_enabled(name)]


@pytest.fixture
def tool_manager() -> ToolManager:
    return make_manager()


class TestToolsList:
    def test_no_arguments_lists_every_tool_with_its_state(self, fake_ui: FakeUI, tool_manager: ToolManager) -> None:
        ToolsCommand(fake_ui, tool_manager).execute()

        assert fake_ui.messages == ["Available tools:\n- echo: enabled\n- weather: enabled"]

    def test_list_argument_lists_every_tool_with_its_state(self, fake_ui: FakeUI, tool_manager: ToolManager) -> None:
        ToolsCommand(fake_ui, tool_manager).execute("list")

        assert fake_ui.messages == ["Available tools:\n- echo: enabled\n- weather: enabled"]

    def test_tools_are_listed_in_alphabetical_order(self, fake_ui: FakeUI) -> None:
        manager = ToolManager([StubTool("zebra"), StubTool("alpha"), StubTool("middle")])

        ToolsCommand(fake_ui, manager).execute("list")

        lines = [line for line in fake_ui.messages[0].splitlines() if line.startswith("- ")]
        assert lines == ["- alpha: enabled", "- middle: enabled", "- zebra: enabled"]

    def test_disabled_tools_are_reported_as_disabled(self, fake_ui: FakeUI) -> None:
        ToolsCommand(fake_ui, make_manager(disabled=("echo",))).execute("list")

        assert fake_ui.messages == ["Available tools:\n- echo: disabled\n- weather: enabled"]

    def test_listing_without_any_registered_tool(self, fake_ui: FakeUI) -> None:
        ToolsCommand(fake_ui, ToolManager()).execute("list")

        assert fake_ui.messages == ["No tools are available."]

    def test_extra_arguments_after_list_are_ignored(self, fake_ui: FakeUI, tool_manager: ToolManager) -> None:
        ToolsCommand(fake_ui, tool_manager).execute("list", "unused", "args")

        assert fake_ui.messages == ["Available tools:\n- echo: enabled\n- weather: enabled"]

    def test_listing_leaves_every_tool_state_untouched(self, fake_ui: FakeUI) -> None:
        manager = make_manager(disabled=("echo",))

        ToolsCommand(fake_ui, manager).execute("list")

        assert enabled_names(manager) == ["weather"]


class TestToolsEnable:
    def test_enables_a_disabled_tool(self, fake_ui: FakeUI) -> None:
        manager = make_manager(disabled=("echo",))

        ToolsCommand(fake_ui, manager).execute("enable", "echo")

        assert manager.is_enabled("echo")
        assert fake_ui.messages == ["Tool 'echo' enabled."]

    def test_enabling_an_already_enabled_tool_keeps_it_enabled(
        self, fake_ui: FakeUI, tool_manager: ToolManager
    ) -> None:
        ToolsCommand(fake_ui, tool_manager).execute("enable", "echo")

        assert enabled_names(tool_manager) == ["echo", "weather"]
        assert fake_ui.messages == ["Tool 'echo' enabled."]

    def test_enabling_only_affects_the_named_tool(self, fake_ui: FakeUI) -> None:
        manager = make_manager(disabled=("echo", "weather"))

        ToolsCommand(fake_ui, manager).execute("enable", "echo")

        assert enabled_names(manager) == ["echo"]

    def test_unknown_tool_name_is_silently_ignored(self, fake_ui: FakeUI, tool_manager: ToolManager) -> None:
        # `_enable_tool` reports nothing on an unknown name, unlike `_disable_tool`.
        ToolsCommand(fake_ui, tool_manager).execute("enable", "ghost")

        assert enabled_names(tool_manager) == ["echo", "weather"]
        assert fake_ui.messages == []

    def test_missing_tool_name_displays_usage(self, fake_ui: FakeUI, tool_manager: ToolManager) -> None:
        ToolsCommand(fake_ui, tool_manager).execute("enable")

        assert fake_ui.messages == [USAGE]

    def test_too_many_arguments_displays_usage(self, fake_ui: FakeUI) -> None:
        manager = make_manager(disabled=("echo",))

        ToolsCommand(fake_ui, manager).execute("enable", "echo", "weather")

        assert not manager.is_enabled("echo")
        assert fake_ui.messages == [USAGE]


class TestToolsDisable:
    def test_disables_an_enabled_tool(self, fake_ui: FakeUI, tool_manager: ToolManager) -> None:
        ToolsCommand(fake_ui, tool_manager).execute("disable", "echo")

        assert not tool_manager.is_enabled("echo")
        assert fake_ui.messages == ["Tool 'echo' disabled."]

    def test_disabling_an_already_disabled_tool_keeps_it_disabled(self, fake_ui: FakeUI) -> None:
        manager = make_manager(disabled=("echo",))

        ToolsCommand(fake_ui, manager).execute("disable", "echo")

        assert enabled_names(manager) == ["weather"]
        assert fake_ui.messages == ["Tool 'echo' disabled."]

    def test_disabling_only_affects_the_named_tool(self, fake_ui: FakeUI, tool_manager: ToolManager) -> None:
        ToolsCommand(fake_ui, tool_manager).execute("disable", "echo")

        assert enabled_names(tool_manager) == ["weather"]

    def test_unknown_tool_name_is_reported_and_changes_nothing(
        self, fake_ui: FakeUI, tool_manager: ToolManager
    ) -> None:
        ToolsCommand(fake_ui, tool_manager).execute("disable", "ghost")

        assert enabled_names(tool_manager) == ["echo", "weather"]
        assert fake_ui.messages == ["Tool 'ghost' is not available."]

    def test_missing_tool_name_displays_usage(self, fake_ui: FakeUI, tool_manager: ToolManager) -> None:
        ToolsCommand(fake_ui, tool_manager).execute("disable")

        assert enabled_names(tool_manager) == ["echo", "weather"]
        assert fake_ui.messages == [USAGE]

    def test_too_many_arguments_displays_usage(self, fake_ui: FakeUI, tool_manager: ToolManager) -> None:
        ToolsCommand(fake_ui, tool_manager).execute("disable", "echo", "weather")

        assert enabled_names(tool_manager) == ["echo", "weather"]
        assert fake_ui.messages == [USAGE]


class TestToolsInvalidAction:
    def test_unknown_action_displays_usage(self, fake_ui: FakeUI, tool_manager: ToolManager) -> None:
        ToolsCommand(fake_ui, tool_manager).execute("bogus")

        assert fake_ui.messages == [USAGE]

    def test_unknown_action_with_a_tool_name_displays_usage(self, fake_ui: FakeUI, tool_manager: ToolManager) -> None:
        ToolsCommand(fake_ui, tool_manager).execute("toggle", "echo")

        assert enabled_names(tool_manager) == ["echo", "weather"]
        assert fake_ui.messages == [USAGE]

    @pytest.mark.parametrize("action", ["LIST", "Enable", "DISABLE"])
    def test_actions_are_case_sensitive(self, fake_ui: FakeUI, tool_manager: ToolManager, action: str) -> None:
        ToolsCommand(fake_ui, tool_manager).execute(action, "echo")

        assert fake_ui.messages == [USAGE]


class TestToolsCommandMetadata:
    def test_registry_metadata_is_exposed(self) -> None:
        assert ToolsCommand.name == "tools"
        assert ToolsCommand.help == "List available tools and toggle them."
        assert ToolsCommand.usage == "/tools [list|enable|disable] [name]"


class TestToolsCompletions:
    def test_no_argument_completes_the_subcommand_names(self, fake_ui: FakeUI, tool_manager: ToolManager) -> None:
        assert ToolsCommand(fake_ui, tool_manager).completions([]) == ["disable", "enable", "list"]

    def test_subcommand_names_are_completed_even_without_any_tool(self, fake_ui: FakeUI) -> None:
        assert ToolsCommand(fake_ui, ToolManager()).completions([]) == ["disable", "enable", "list"]

    @pytest.mark.parametrize("subcommand", ["enable", "disable"])
    def test_toggling_subcommands_complete_the_tool_names(self, fake_ui: FakeUI, subcommand: str) -> None:
        manager = ToolManager([StubTool("zebra"), StubTool("alpha"), StubTool("middle")])

        assert ToolsCommand(fake_ui, manager).completions([subcommand]) == ["alpha", "middle", "zebra"]

    def test_disabled_tools_are_completed_too(self, fake_ui: FakeUI) -> None:
        # `/tools enable <name>` must be able to complete a name that is currently disabled.
        manager = make_manager(disabled=("echo",))

        assert ToolsCommand(fake_ui, manager).completions(["enable"]) == ["echo", "weather"]

    def test_no_tool_name_is_completed_without_any_registered_tool(self, fake_ui: FakeUI) -> None:
        assert ToolsCommand(fake_ui, ToolManager()).completions(["enable"]) == []

    def test_list_subcommand_takes_no_argument(self, fake_ui: FakeUI, tool_manager: ToolManager) -> None:
        assert ToolsCommand(fake_ui, tool_manager).completions(["list"]) == []

    def test_unknown_subcommand_completes_nothing(self, fake_ui: FakeUI, tool_manager: ToolManager) -> None:
        assert ToolsCommand(fake_ui, tool_manager).completions(["bogus"]) == []

    @pytest.mark.parametrize("args", [["enable", "echo"], ["disable", "echo"], ["list", "echo"]])
    def test_nothing_is_completed_past_the_tool_name(
        self, fake_ui: FakeUI, tool_manager: ToolManager, args: list[str]
    ) -> None:
        assert ToolsCommand(fake_ui, tool_manager).completions(args) == []

    def test_completing_reports_nothing_to_the_ui(self, fake_ui: FakeUI, tool_manager: ToolManager) -> None:
        ToolsCommand(fake_ui, tool_manager).completions([])

        assert fake_ui.messages == []

    def test_completing_leaves_every_tool_state_untouched(self, fake_ui: FakeUI) -> None:
        manager = make_manager(disabled=("echo",))

        ToolsCommand(fake_ui, manager).completions(["enable"])

        assert enabled_names(manager) == ["weather"]


class TestToolsCompletionMatchesExecute:
    @pytest.mark.parametrize("subcommand", ["list", "enable", "disable"])
    def test_every_completed_subcommand_is_accepted_by_execute(
        self, fake_ui: FakeUI, tool_manager: ToolManager, subcommand: str
    ) -> None:
        command = ToolsCommand(fake_ui, tool_manager)

        assert subcommand in command.completions([])
        command.execute(subcommand, "echo")

        assert fake_ui.messages != [USAGE]

    @pytest.mark.parametrize("subcommand", ["set", "remove"])
    def test_unknown_subcommand_names_are_rejected(
        self, fake_ui: FakeUI, tool_manager: ToolManager, subcommand: str
    ) -> None:
        ToolsCommand(fake_ui, tool_manager).execute(subcommand, "echo")

        assert enabled_names(tool_manager) == ["echo", "weather"]
        assert fake_ui.messages == [USAGE]

    def test_usage_names_exactly_the_completed_subcommands(self, fake_ui: FakeUI, tool_manager: ToolManager) -> None:
        command = ToolsCommand(fake_ui, tool_manager)

        assert command.usage == "/tools [list|enable|disable] [name]"
        assert command.completions([]) == ["disable", "enable", "list"]


class TestToolsEffectOnModelPayload:
    def test_disabling_a_tool_removes_its_rule_from_the_payload(
        self, fake_ui: FakeUI, tool_manager: ToolManager
    ) -> None:
        ToolsCommand(fake_ui, tool_manager).execute("disable", "echo")

        assert tool_manager.get_tool_call_rules() == [{"type": "function", "function": {"name": "weather"}}]

    def test_enabling_a_tool_restores_its_rule_in_the_payload(self, fake_ui: FakeUI) -> None:
        manager = make_manager(disabled=("echo", "weather"))

        ToolsCommand(fake_ui, manager).execute("enable", "weather")

        assert manager.get_tool_call_rules() == [{"type": "function", "function": {"name": "weather"}}]

    def test_disabled_tool_becomes_unreachable_for_the_model(self, fake_ui: FakeUI, tool_manager: ToolManager) -> None:
        call = {"type": "function", "function": {"name": "echo", "arguments": {}}}

        ToolsCommand(fake_ui, tool_manager).execute("disable", "echo")

        assert tool_manager.call(call) is None


class TestToolsResponseAppending:
    def test_enable_message_is_appended_to_the_response(self) -> None:
        ui = RecordingUI()

        ToolsCommand(ui, make_manager(disabled=("echo",))).execute("enable", "echo")

        assert ui.appended == [("Tool 'echo' enabled.", True)]

    def test_disable_message_is_appended_to_the_response(self) -> None:
        ui = RecordingUI()

        ToolsCommand(ui, make_manager()).execute("disable", "echo")

        assert ui.appended == [("Tool 'echo' disabled.", True)]

    def test_listing_is_not_appended_to_the_response(self) -> None:
        ui = RecordingUI()

        ToolsCommand(ui, make_manager()).execute("list")

        assert ui.appended == [("Available tools:\n- echo: enabled\n- weather: enabled", False)]

    def test_unknown_tool_message_is_not_appended_to_the_response(self) -> None:
        ui = RecordingUI()

        ToolsCommand(ui, make_manager()).execute("disable", "ghost")

        assert ui.appended == [("Tool 'ghost' is not available.", False)]

    def test_enabling_an_unknown_tool_sends_no_message_at_all(self) -> None:
        ui = RecordingUI()

        ToolsCommand(ui, make_manager()).execute("enable", "ghost")

        assert ui.appended == []

    def test_usage_message_is_not_appended_to_the_response(self) -> None:
        ui = RecordingUI()

        ToolsCommand(ui, make_manager()).execute("enable")

        assert ui.appended == [(USAGE, False)]
