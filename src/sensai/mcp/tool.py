from __future__ import annotations

from typing import TYPE_CHECKING

from sensai.tools.tool import Tool

if TYPE_CHECKING:
    from sensai.mcp.discovery import MCPToolInfo


class MCPTool(Tool[object]):
    """Expose an MCP tool through Sensai's existing tool contract."""

    def __init__(self, info: MCPToolInfo) -> None:
        """Store the discovered MCP tool metadata."""
        self._info = info

    @property
    def name(self) -> str:
        """Return the name advertised by the MCP server."""
        return self._info.name

    @property
    def tool_rule(self) -> dict[str, object]:
        """Return the tool schema in the format expected by the model provider."""
        function: dict[str, object] = {
            "name": self._info.name,
            "parameters": self._info.input_schema,
        }
        if self._info.description is not None:
            function["description"] = self._info.description
        return {"type": "function", "function": function}

    def run(self, **kwargs: object) -> object:
        """Execute the MCP tool once MCP call support is implemented."""
        del kwargs
        raise NotImplementedError
