from __future__ import annotations

from typing import TYPE_CHECKING

import anyio

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


async def start_clients(clients: list[MCPClient]) -> list[MCPTool]:
    """Start every client concurrently and return their tools, in the order of `clients`."""
    tools: list[list[MCPTool]] = [[] for _ in clients]

    async def start(index: int, client: MCPClient) -> None:
        tools[index] = await client.start()

    try:
        async with anyio.create_task_group() as task_group:
            for index, client in enumerate(clients):
                task_group.start_soon(start, index, client)
    except ExceptionGroup as group:
        # Raise the first failure itself, so a server that cannot start still stops Sensai with its own error.
        raise group.exceptions[0] from None
    return [tool for client_tools in tools for tool in client_tools]
