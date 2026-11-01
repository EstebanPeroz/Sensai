from unittest.mock import MagicMock

import pytest

from sensai.mcp.discovery import MCPToolInfo
from sensai.mcp.errors import MCPCallError
from sensai.mcp.tool import MCPTool
from sensai.tools.tool_manager import ToolManager


def make_tool(session: MagicMock | None = None) -> MCPTool:
    if session is None:
        session = MagicMock(server_name="server")
    return MCPTool(
        MCPToolInfo(
            name="echo",
            description="Echo a message",
            input_schema={"type": "object", "properties": {"message": {"type": "string"}}},
        ),
        session,
    )


class TestMCPTool:
    def test_name_is_the_discovered_tool_name_prefixed_by_its_server(self) -> None:
        assert make_tool().name == "server__echo"

    def test_same_tool_of_two_servers_gets_two_names(self) -> None:
        first = make_tool(MagicMock(server_name="first"))
        second = make_tool(MagicMock(server_name="second"))

        assert ToolManager([first, second]).names() == ["first__echo", "second__echo"]

    def test_tool_rule_matches_the_existing_function_schema(self) -> None:
        assert make_tool().tool_rule == {
            "type": "function",
            "function": {
                "name": "server__echo",
                "description": "Echo a message",
                "parameters": {"type": "object", "properties": {"message": {"type": "string"}}},
            },
        }

    def test_run_calls_the_server_with_the_unprefixed_name_and_returns_text(self) -> None:
        session = MagicMock(server_name="server")
        session.call_tool_sync.return_value = "Echoed message"
        tool = make_tool(session)

        assert tool.run(message="hello") == "Echoed message"
        session.call_tool_sync.assert_called_once_with("echo", {"message": "hello"})

    def test_run_propagates_mcp_call_errors(self) -> None:
        session = MagicMock()
        session.call_tool_sync.side_effect = MCPCallError("tool failed")
        tool = make_tool(session)

        with pytest.raises(MCPCallError, match="tool failed"):
            tool.run(message="hello")
