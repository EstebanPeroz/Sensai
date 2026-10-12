from sensai.mcp.discovery import MCPToolInfo
from sensai.mcp.tool import MCPTool


def make_tool() -> MCPTool:
    return MCPTool(
        MCPToolInfo(
            name="echo",
            description="Echo a message",
            input_schema={"type": "object", "properties": {"message": {"type": "string"}}},
        ),
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
