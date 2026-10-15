import pytest

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


class TestGetToolCallRules:
    def test_returns_the_rule_of_every_tool(self) -> None:
        assert make_manager().get_tool_call_rules() == [
            EchoTool().tool_rule,
            AddTool().tool_rule,
            ProfileTool().tool_rule,
        ]

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

    def test_manager_without_tools_has_no_rules(self) -> None:
        assert ToolManager().get_tool_call_rules() == []
