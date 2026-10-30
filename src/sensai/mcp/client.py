from __future__ import annotations

from typing import TYPE_CHECKING

import anyio

from sensai.mcp.errors import MCPError, MCPStartupError
from sensai.mcp.session import MCPServerSession
from sensai.mcp.tool import MCPTool

if TYPE_CHECKING:
    from sensai.mcp.config import MCPServerConfig


class MCPClient:
    """Create tools from one configured MCP server."""

    def __init__(self, config: MCPServerConfig) -> None:
        """Store the server configuration without starting it."""
        self._config = config

    @property
    def name(self) -> str:
        """Return the configured name of the server."""
        return self._config.name

    async def start(self) -> list[MCPTool]:
        """Start the configured server and return its discovered tools.

        Raises:
            MCPError: If the server cannot start, or does not list its tools within the startup timeout.

        """
        if not self._config.enabled:
            return []

        session = MCPServerSession(self._config)
        await session.start()
        try:
            with anyio.fail_after(self._config.startup_timeout):
                tools = await session.list_tools()
        except Exception as err:
            await session.close()
            raise MCPStartupError.from_error(err, self._config.startup_timeout) from err
        await session.close()
        return [MCPTool(info, session) for info in tools]


async def start_clients(clients: list[MCPClient]) -> tuple[list[MCPTool], dict[str, MCPError]]:
    """Start every client concurrently, skipping the ones that fail.

    Returns the tools of the started clients, in the order of `clients`, and the error of each failed client
    by server name, so that one unavailable server does not take the others down with it.
    """
    tools: list[list[MCPTool]] = [[] for _ in clients]
    failures: dict[str, MCPError] = {}

    async def start(index: int, client: MCPClient) -> None:
        try:
            tools[index] = await client.start()
        except MCPError as err:
            failures[client.name] = err

    async with anyio.create_task_group() as task_group:
        for index, client in enumerate(clients):
            task_group.start_soon(start, index, client)
    return [tool for client_tools in tools for tool in client_tools], failures
