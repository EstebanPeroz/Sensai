from unittest.mock import MagicMock

from sensai.mcp.discovery import MCPToolInfo
from sensai.mcp.request import add_tool_result, add_tool_schemas, execute_one_tool_call
from sensai.mcp.tool import MCPTool
from sensai.tools.tool_manager import ToolManager


def test_add_tool_schemas_adds_enabled_schemas() -> None:
    payload: dict = {"model": "test", "messages": []}
    tool_manager = type("ToolManager", (), {"get_tool_call_rules": lambda _: [{"type": "function"}]})()

    add_tool_schemas(payload, tool_manager)

    assert payload == {
        "model": "test",
        "messages": [],
        "tools": [{"type": "function"}],
    }


def test_add_tool_schemas_leaves_payload_unchanged_without_tools() -> None:
    payload: dict = {"model": "test", "messages": []}
    tool_manager = type("ToolManager", (), {"get_tool_call_rules": lambda _: []})()

    add_tool_schemas(payload, tool_manager)

    assert payload == {"model": "test", "messages": []}


def test_execute_one_tool_call_returns_the_first_tool_result() -> None:
    result = {"role": "tool", "tool_name": "echo", "content": "hello"}
    tool_manager = type("ToolManager", (), {"call": lambda *_: result})()

    assert execute_one_tool_call([{"function": {"name": "echo"}}], tool_manager) == result


def test_execute_one_tool_call_does_not_call_manager_without_calls() -> None:
    class UnusedToolManager:
        def call(self, tool_call: dict) -> dict:
            del tool_call
            raise AssertionError

    assert execute_one_tool_call([], UnusedToolManager()) is None


def test_add_tool_result_builds_one_follow_up_request() -> None:
    payload = {"model": "test", "messages": [{"role": "user", "content": "say hello"}], "tools": [{"type": "function"}]}
    tool_call = {"type": "function", "function": {"name": "echo", "arguments": {"message": "hello"}}}
    tool_result = {"role": "tool", "tool_name": "echo", "content": "hello from MCP"}

    assert add_tool_result(payload, tool_call, tool_result) == {
        "model": "test",
        "messages": [
            {"role": "user", "content": "say hello"},
            {"role": "assistant", "tool_calls": [tool_call]},
            {"role": "tool", "content": "hello from MCP"},
        ],
        "tools": [{"type": "function"}],
    }
    assert payload["messages"] == [{"role": "user", "content": "say hello"}]


def test_model_tool_call_reaches_mcp_server() -> None:
    session = MagicMock()
    session.call_tool_sync.return_value = "hello from the MCP server"
    tool = MCPTool(
        MCPToolInfo("echo", "Echo text", {"type": "object"}),
        session,
    )

    result = execute_one_tool_call(
        [
            {
                "type": "function",
                "function": {"name": "echo", "arguments": {"message": "hello from the model"}},
            },
        ],
        ToolManager([tool]),
    )

    assert result == {
        "role": "tool",
        "tool_name": "echo",
        "content": "hello from the MCP server",
    }
    session.call_tool_sync.assert_called_once_with("echo", {"message": "hello from the model"})
