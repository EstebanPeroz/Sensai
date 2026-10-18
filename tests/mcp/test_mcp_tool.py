from unittest.mock import MagicMock

import pytest

from sensai.mcp.discovery import MCPToolInfo
from sensai.mcp.errors import MCPCallError
from sensai.mcp.tool import MCPTool


def make_tool(session: MagicMock | None = None) -> MCPTool:
    if session is None:
        session = MagicMock()
    return MCPTool(
        MCPToolInfo(
            name="echo",
            description="Echo a message",
            input_schema={"type": "object", "properties": {"message": {"type": "string"}}},
        ),
        session,
    )


class TestMCPTool:
    def test_name_matches_the_discovered_tool_name(self) -> None:
        assert make_tool().name == "echo"

    def test_tool_rule_matches_the_existing_function_schema(self) -> None:
        assert make_tool().tool_rule == {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo a message",
                "parameters": {"type": "object", "properties": {"message": {"type": "string"}}},
            },
        }

    def test_run_calls_the_owning_session_and_returns_text(self) -> None:
        session = MagicMock()
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
