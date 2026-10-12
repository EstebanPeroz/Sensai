from __future__ import annotations

from typing import TYPE_CHECKING

import anyio

from sensai.tools.tool import Tool

if TYPE_CHECKING:
    from sensai.mcp.discovery import MCPToolInfo
    from sensai.mcp.session import MCPServerSession


class MCPTool(Tool[object]):
    """Expose an MCP tool through Sensai's existing tool contract."""

    def __init__(self, info: MCPToolInfo, session: MCPServerSession) -> None:
        """Store the discovered MCP tool metadata."""
        self._info = info
        self._session = session

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

    def run(self, **kwargs: object) -> str:
        """Execute the MCP tool and return its simple text result."""
        return anyio.run(self._session.call_tool, self.name, kwargs)
