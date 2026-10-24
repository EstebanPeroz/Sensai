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

    async def start(self) -> list[MCPTool]:
        """Start the configured server and return its discovered tools."""
        if not self._config.enabled:
            return []

        session = MCPServerSession(self._config)
        await session.start()
        try:
            tools = await session.list_tools()
        except Exception:
            await session.close()
            raise
        await session.close()
        return [MCPTool(info, session) for info in tools]
