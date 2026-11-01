from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from sensai_server.tools.tool import Tool, ToolContext

if TYPE_CHECKING:
    from sensai_server.tools.mcp.discovery import MCPToolInfo
    from sensai_server.tools.mcp.host import MCPHost
    from sensai_server.tools.mcp.session import MCPServerSession


class MCPTool(Tool[str]):
    """Expose a tool of an MCP server as a server `Tool`, shared by every session like the built-in ones."""

    def __init__(self, info: MCPToolInfo, session: MCPServerSession, host: MCPHost) -> None:
        """Store the discovered tool, the session of its server, and the host that runs that session."""
        self._info = info
        self._session = session
        self._host = host

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

    async def run(self, ctx: ToolContext, **kwargs: object) -> str:  # noqa: ARG002 - an MCP tool never talks to the client
        """Call the tool on the host's loop, without blocking this one, and return its text result.

        Cancelling this call cancels the MCP call on the host's loop.
        """
        return await asyncio.wrap_future(self._host.call(self._session, self._info.name, kwargs))
