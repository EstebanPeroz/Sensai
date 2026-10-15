import pytest

from sensai.error import SensaiError
from sensai.tools.error import DuplicateToolError
from sensai.tools.tool import Tool, get_parameter
from sensai.tools.tool_manager import ToolManager


class EchoTool(Tool[str]):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def tool_rule(self) -> dict:
        return {"type": "function", "function": {"name": self.name}}

    def run(self, **kwargs: object) -> str:
        return get_parameter("city", str, **kwargs)


class NamedTool(Tool[str]):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def tool_rule(self) -> dict:
        return {"type": "function", "function": {"name": self._name}}

    def run(self, **kwargs: object) -> str:
        del kwargs
        return f"{self._name} ran"


def make_manager() -> ToolManager:
    return ToolManager([EchoTool()])


def make_pair(disabled: tuple[str, ...] = ()) -> ToolManager:
    return ToolManager([NamedTool("weather"), NamedTool("echo")], disabled=disabled)


def enabled_names(manager: ToolManager) -> list[str]:
    return [name for name in manager.names() if manager.is_enabled(name)]


def make_call(name: str, arguments: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "index": 0,
            "name": name,
            "arguments": arguments or {},
        },
    }


class TestCall:
    def test_calls_matching_tool_with_arguments(self) -> None:
        result = make_manager().call(make_call("echo", {"city": "Paris"}))

        assert result == {"role": "tool", "tool_name": "echo", "content": "Paris"}

    def test_unknown_tool_name_returns_none(self) -> None:
        assert make_manager().call(make_call("does-not-exist")) is None

    def test_missing_function_key_returns_empty_dict(self) -> None:
        assert make_manager().call({"type": "function"}) == {}

    def test_missing_name_returns_empty_dict(self) -> None:
        assert make_manager().call({"type": "function", "function": {"arguments": {}}}) == {}

    def test_wrong_argument_type_propagates_tool_error(self) -> None:
        with pytest.raises(SensaiError, match="Parameter 'city' must be str, got int"):
            make_manager().call(make_call("echo", {"city": 2}))

    def test_missing_argument_propagates_tool_error(self) -> None:
        with pytest.raises(SensaiError, match="Miss parameter: city"):
            make_manager().call(make_call("echo"))


class TestRegistration:
    def test_names_are_returned_sorted(self) -> None:
        manager = ToolManager([NamedTool("zebra"), NamedTool("alpha"), NamedTool("middle")])

        assert manager.names() == ["alpha", "middle", "zebra"]

    def test_names_include_disabled_tools(self) -> None:
        assert make_pair(disabled=("echo",)).names() == ["echo", "weather"]

    def test_duplicate_names_are_rejected(self) -> None:
        with pytest.raises(DuplicateToolError, match="two tools are registered as 'echo'"):
            ToolManager([NamedTool("echo"), NamedTool("echo")])

    def test_duplicate_error_is_a_sensai_error(self) -> None:
        with pytest.raises(SensaiError):
            ToolManager([NamedTool("echo"), NamedTool("echo")])

    def test_manager_without_arguments_registers_nothing(self) -> None:
        manager = ToolManager()

        assert manager.names() == []
        assert manager.get_tool_call_rules() == []

    def test_explicit_none_arguments_register_nothing(self) -> None:
        assert ToolManager(None, None).names() == []


class TestDisabledAtStartup:
    def test_every_tool_is_enabled_by_default(self) -> None:
        assert enabled_names(make_pair()) == ["echo", "weather"]

    def test_named_tools_start_disabled(self) -> None:
        assert enabled_names(make_pair(disabled=("echo",))) == ["weather"]

    def test_all_tools_can_start_disabled(self) -> None:
        assert enabled_names(make_pair(disabled=("echo", "weather"))) == []

    def test_unknown_disabled_names_are_ignored(self) -> None:
        assert enabled_names(make_pair(disabled=("ghost",))) == ["echo", "weather"]

    def test_unknown_disabled_names_are_reported_sorted(self) -> None:
        assert make_pair(disabled=("zzz", "ghost")).unknown_disabled() == ["ghost", "zzz"]

    def test_unknown_disabled_is_empty_when_every_name_matches(self) -> None:
        assert make_pair(disabled=("echo",)).unknown_disabled() == []

    def test_disabled_accepts_any_iterable(self) -> None:
        manager = ToolManager([NamedTool("weather"), NamedTool("echo")], disabled=(n for n in ["echo"]))

        assert enabled_names(manager) == ["weather"]


class TestIsEnabled:
    def test_unknown_name_is_not_enabled(self) -> None:
        assert make_pair().is_enabled("ghost") is False

    def test_registered_tool_is_enabled(self) -> None:
        assert make_pair().is_enabled("echo") is True


class TestEnableAndDisable:
    def test_disable_then_enable_restores_the_tool(self) -> None:
        manager = make_pair()

        assert manager.disable("echo") is True
        assert manager.is_enabled("echo") is False
        assert manager.enable("echo") is True
        assert manager.is_enabled("echo") is True

    def test_enable_is_idempotent(self) -> None:
        manager = make_pair()

        assert manager.enable("echo") is True
        assert manager.enable("echo") is True
        assert enabled_names(manager) == ["echo", "weather"]

    def test_disable_is_idempotent(self) -> None:
        manager = make_pair()

        assert manager.disable("echo") is True
        assert manager.disable("echo") is True
        assert enabled_names(manager) == ["weather"]

    def test_enable_rejects_an_unknown_name(self) -> None:
        manager = make_pair()

        assert manager.enable("ghost") is False
        assert manager.names() == ["echo", "weather"]

    def test_disable_rejects_an_unknown_name(self) -> None:
        manager = make_pair()

        assert manager.disable("ghost") is False
        assert enabled_names(manager) == ["echo", "weather"]

    def test_a_name_given_as_disabled_but_unknown_stays_unknown(self) -> None:
        manager = make_pair(disabled=("ghost",))

        assert manager.enable("ghost") is False

    def test_toggling_one_tool_leaves_the_others_untouched(self) -> None:
        manager = make_pair()

        manager.disable("echo")

        assert manager.is_enabled("weather") is True


class TestGetToolCallRules:
    def test_returns_the_rule_of_every_enabled_tool(self) -> None:
        assert make_pair().get_tool_call_rules() == [
            {"type": "function", "function": {"name": "echo"}},
            {"type": "function", "function": {"name": "weather"}},
        ]

    def test_rules_are_sorted_by_tool_name(self) -> None:
        manager = ToolManager([NamedTool("zebra"), NamedTool("alpha")])

        assert [rule["function"]["name"] for rule in manager.get_tool_call_rules()] == ["alpha", "zebra"]

    def test_disabled_tools_are_left_out(self) -> None:
        assert make_pair(disabled=("echo",)).get_tool_call_rules() == [
            {"type": "function", "function": {"name": "weather"}}
        ]

    def test_no_rule_is_returned_when_every_tool_is_disabled(self) -> None:
        assert make_pair(disabled=("echo", "weather")).get_tool_call_rules() == []

    def test_rules_follow_a_tool_being_disabled(self) -> None:
        manager = make_pair()

        manager.disable("echo")

        assert manager.get_tool_call_rules() == [{"type": "function", "function": {"name": "weather"}}]


class TestCallOnDisabledTool:
    def test_disabled_tool_is_unreachable(self) -> None:
        manager = ToolManager([EchoTool()], disabled=("echo",))

        assert manager.call(make_call("echo", {"city": "Paris"})) is None

    def test_tool_disabled_after_registration_is_unreachable(self) -> None:
        manager = make_manager()

        manager.disable("echo")

        assert manager.call(make_call("echo", {"city": "Paris"})) is None

    def test_re_enabled_tool_is_reachable_again(self) -> None:
        manager = ToolManager([EchoTool()], disabled=("echo",))

        manager.enable("echo")

        assert manager.call(make_call("echo", {"city": "Paris"})) == {
            "role": "tool",
            "tool_name": "echo",
            "content": "Paris",
        }

    def test_disabled_tool_is_not_run(self) -> None:
        manager = ToolManager([EchoTool()], disabled=("echo",))

        # A missing `city` would raise if the tool ran; being disabled must short-circuit first.
        assert manager.call(make_call("echo")) is None

    def test_manager_without_tools_reaches_nothing(self) -> None:
        assert ToolManager().call(make_call("echo", {"city": "Paris"})) is None
