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
    def description(self) -> str:
        return "Return the given city."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}

    def run(self, **kwargs: object) -> str:
        return get_parameter("city", str, **kwargs)


class AddTool(Tool[int]):
    @property
    def name(self) -> str:
        return "add"

    @property
    def description(self) -> str:
        return "Add two integers."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }

    def run(self, a: int, b: int) -> int:  # type: ignore[override]
        return a + b


class ProfileTool(Tool[dict]):
    @property
    def name(self) -> str:
        return "profile"

    @property
    def description(self) -> str:
        return "Return a fixed profile."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def run(self, **kwargs: object) -> dict:
        del kwargs
        return {"city": "Paris", "age": 30}


class BrokenTool(Tool[str]):
    @property
    def name(self) -> str:
        return "broken"

    @property
    def description(self) -> str:
        return "Always fails."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def run(self, **kwargs: object) -> str:
        del kwargs
        msg = "page could not be parsed"
        raise ValueError(msg)


class InterruptedTool(BrokenTool):
    @property
    def name(self) -> str:
        return "interrupted"

    def run(self, **kwargs: object) -> str:
        del kwargs
        raise KeyboardInterrupt


class NamedTool(Tool[str]):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"The {self._name} tool."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def run(self, **kwargs: object) -> str:
        del kwargs
        return f"{self._name} ran"


def make_manager() -> ToolManager:
    return ToolManager([EchoTool(), AddTool(), ProfileTool()])


def make_call(name: str, arguments: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "index": 0,
            "name": name,
            "arguments": arguments or {},
        },
    }


def make_pair(disabled: tuple[str, ...] = ()) -> ToolManager:
    return ToolManager([NamedTool("weather"), NamedTool("echo")], disabled=disabled)


def enabled_names(manager: ToolManager) -> list[str]:
    return [name for name in manager.names() if manager.is_enabled(name)]


def rule_names(manager: ToolManager) -> list[str]:
    return [rule["function"]["name"] for rule in manager.get_tool_call_rules()]


class TestCall:
    def test_calls_matching_tool_with_arguments(self) -> None:
        result = make_manager().call(make_call("echo", {"city": "Paris"}))

        assert result == {"role": "tool", "tool_name": "echo", "content": "Paris"}

    def test_unknown_tool_returns_error_message(self) -> None:
        result = make_manager().call(make_call("does-not-exist"))

        assert result == {
            "role": "tool",
            "tool_name": "does-not-exist",
            "content": "error: unknown tool 'does-not-exist'",
        }

    def test_missing_function_key_returns_error_message(self) -> None:
        result = make_manager().call({"type": "function"})

        assert result == {
            "role": "tool",
            "tool_name": "unknown",
            "content": "error: malformed tool call, missing function name",
        }

    def test_missing_name_returns_error_message(self) -> None:
        result = make_manager().call({"type": "function", "function": {"arguments": {}}})

        assert result == {
            "role": "tool",
            "tool_name": "unknown",
            "content": "error: malformed tool call, missing function name",
        }

    @pytest.mark.parametrize("function", ["web_search", ["web_search"], 3])
    def test_function_that_is_not_an_object_returns_error_message(self, function: object) -> None:
        result = make_manager().call({"type": "function", "function": function})

        assert result == {
            "role": "tool",
            "tool_name": "unknown",
            "content": "error: malformed tool call, function must be an object",
        }

    def test_call_that_is_not_an_object_returns_error_message(self) -> None:
        result = make_manager().call("web_search")  # type: ignore[arg-type]

        assert result == {
            "role": "tool",
            "tool_name": "unknown",
            "content": "error: malformed tool call, missing function name",
        }

    def test_wrong_argument_type_returns_error_message(self) -> None:
        result = make_manager().call(make_call("echo", {"city": 2}))

        assert result == {
            "role": "tool",
            "tool_name": "echo",
            "content": "error: Parameter 'city' must be str, got int",
        }

    def test_missing_argument_returns_error_message(self) -> None:
        result = make_manager().call(make_call("echo"))

        assert result == {"role": "tool", "tool_name": "echo", "content": "error: Miss parameter: city"}

    def test_missing_arguments_are_treated_as_empty(self) -> None:
        result = make_manager().call({"type": "function", "function": {"name": "echo", "arguments": None}})

        assert result == {"role": "tool", "tool_name": "echo", "content": "error: Miss parameter: city"}

    def test_unexpected_argument_returns_error_message(self) -> None:
        result = make_manager().call(make_call("add", {"a": 1, "b": 2, "c": 3}))

        assert result["role"] == "tool"
        assert result["tool_name"] == "add"
        assert result["content"].startswith("error: ")
        assert "unexpected keyword argument 'c'" in result["content"]

    def test_number_result_is_converted_to_text(self) -> None:
        result = make_manager().call(make_call("add", {"a": 1, "b": 2}))

        assert result == {"role": "tool", "tool_name": "add", "content": "3"}

    def test_dict_result_is_converted_to_json_text(self) -> None:
        result = make_manager().call(make_call("profile"))

        assert result == {"role": "tool", "tool_name": "profile", "content": '{"city": "Paris", "age": 30}'}


class TestToolName:
    def test_reads_the_name_of_a_well_formed_call(self) -> None:
        assert ToolManager.tool_name(make_call("echo")) == "echo"

    @pytest.mark.parametrize(
        "tool_call",
        [
            "web_search",
            {"type": "function"},
            {"type": "function", "function": "web_search"},
            {"type": "function", "function": {"arguments": {}}},
        ],
    )
    def test_is_unknown_for_a_malformed_call(self, tool_call: object) -> None:
        assert ToolManager.tool_name(tool_call) == "unknown"


class TestUnexpectedFailure:
    def test_unexpected_exception_returns_error_message(self) -> None:
        result = ToolManager([BrokenTool()]).call(make_call("broken"))

        assert result == {
            "role": "tool",
            "tool_name": "broken",
            "content": "error: ValueError: page could not be parsed",
        }

    def test_unexpected_exception_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        ToolManager([BrokenTool()]).call(make_call("broken"))

        assert "Tool 'broken' failed unexpectedly" in caplog.text

    def test_interruption_is_not_swallowed(self) -> None:
        with pytest.raises(KeyboardInterrupt):
            ToolManager([InterruptedTool()]).call(make_call("interrupted"))


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
        assert rule_names(make_pair()) == ["echo", "weather"]

    def test_rules_are_sorted_by_tool_name(self) -> None:
        manager = ToolManager([NamedTool("zebra"), NamedTool("alpha")])

        assert rule_names(manager) == ["alpha", "zebra"]

    def test_disabled_tools_are_left_out(self) -> None:
        assert rule_names(make_pair(disabled=("echo",))) == ["weather"]

    def test_no_rule_is_returned_when_every_tool_is_disabled(self) -> None:
        assert make_pair(disabled=("echo", "weather")).get_tool_call_rules() == []

    def test_rules_follow_a_tool_being_disabled(self) -> None:
        manager = make_pair()

        manager.disable("echo")

        assert rule_names(manager) == ["weather"]

    def test_rules_follow_the_chat_endpoint_format(self) -> None:
        assert ToolManager([EchoTool()]).get_tool_call_rules() == [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Return the given city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            },
        ]


class TestCallOnDisabledTool:
    def test_disabled_tool_is_unreachable(self) -> None:
        manager = ToolManager([EchoTool()], disabled=("echo",))

        assert manager.call(make_call("echo", {"city": "Paris"})) == {
            "role": "tool",
            "tool_name": "echo",
            "content": "error: unknown tool 'echo'",
        }

    def test_tool_disabled_after_registration_is_unreachable(self) -> None:
        manager = make_manager()

        manager.disable("echo")

        assert manager.call(make_call("echo", {"city": "Paris"})) == {
            "role": "tool",
            "tool_name": "echo",
            "content": "error: unknown tool 'echo'",
        }

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
        assert manager.call(make_call("echo")) == {
            "role": "tool",
            "tool_name": "echo",
            "content": "error: unknown tool 'echo'",
        }

    def test_manager_without_tools_reaches_nothing(self) -> None:
        assert ToolManager().call(make_call("echo", {"city": "Paris"})) == {
            "role": "tool",
            "tool_name": "echo",
            "content": "error: unknown tool 'echo'",
        }
