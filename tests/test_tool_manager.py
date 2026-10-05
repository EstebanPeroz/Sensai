import pytest

from sensai.error import SensaiError
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


def make_manager() -> ToolManager:
    return ToolManager([EchoTool()])


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
