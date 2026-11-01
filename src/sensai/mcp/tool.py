from __future__ import annotations

from typing import TYPE_CHECKING

from sensai.tools.tool import Tool

if TYPE_CHECKING:
    from sensai.mcp.discovery import MCPToolInfo
    from sensai.mcp.session import MCPServerSession


class MCPTool(Tool[str]):
    """Expose an MCP tool through Sensai's existing tool contract."""

    def __init__(self, info: MCPToolInfo, session: MCPServerSession) -> None:
        """Store the discovered MCP tool metadata."""
        self._info = info
        self._session = session

    @property
    def name(self) -> str:
        """Return the advertised name, prefixed by the server name so that two servers' tools never collide."""
        return f"{self._session.server_name}__{self._info.name}"

    @property
    def description(self) -> str:
        """Return the description advertised by the MCP server."""
        return self._info.description or ""

    @property
    def parameters(self) -> dict[str, object]:
        """Return the input schema advertised by the MCP server."""
        return self._info.input_schema

    def run(self, **kwargs: object) -> str:
        """Execute the MCP tool and return its simple text result."""
        return self._session.call_tool_sync(self._info.name, kwargs)
