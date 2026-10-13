from __future__ import annotations

from typing import TYPE_CHECKING

from sensai.mcp.session import MCPServerSession
from sensai.mcp.tool import MCPTool

if TYPE_CHECKING:
    from sensai.mcp.config import MCPServerConfig


class MCPClient:
    """Create tools from one configured MCP server."""

    def __init__(self, config: MCPServerConfig) -> None:
        """Store the server configuration without starting it."""
        self._config = config
        self._session: MCPServerSession | None = None

    async def start(self) -> list[MCPTool]:
        """Start the configured server and return its discovered tools."""
        if not self._config.enabled:
            return []

        session = MCPServerSession(self._config)
        await session.start()
        self._session = session
        return [MCPTool(info, session) for info in await session.list_tools()]

    async def close(self) -> None:
        """Close the started MCP server, if any."""
        if self._session is None:
            return
        await self._session.close()
        self._session = None
