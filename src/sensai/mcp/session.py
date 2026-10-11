from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from sensai.mcp.discovery import MCPToolInfo
from sensai.mcp.errors import MCPProtocolError, MCPShutdownError, MCPStartupError

if TYPE_CHECKING:
    from sensai.mcp.config import MCPServerConfig


class MCPServerSession:
    """Manage one initialized MCP server session over stdio."""

    def __init__(self, config: MCPServerConfig) -> None:
        """Store the server configuration without starting the process."""
        self._config = config
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def start(self) -> None:
        """Start the configured server and initialize its MCP session."""
        if self._session is not None:
            return

        exit_stack = AsyncExitStack()
        try:
            with anyio.fail_after(self._config.startup_timeout):
                streams = await exit_stack.enter_async_context(
                    stdio_client(
                        StdioServerParameters(
                            command=self._config.command,
                            args=self._config.args,
                        ),
                    ),
                )
                session = await exit_stack.enter_async_context(ClientSession(*streams))
                await session.initialize()
        except Exception as err:
            await exit_stack.aclose()
            raise MCPStartupError(str(err)) from err

        self._exit_stack = exit_stack
        self._session = session

    async def close(self) -> None:
        """Close the MCP session and its server process."""
        if self._exit_stack is None:
            return

        try:
            await self._exit_stack.aclose()
        except Exception as err:
            raise MCPShutdownError(str(err)) from err
        finally:
            self._exit_stack = None
            self._session = None

    async def list_tools(self) -> list[MCPToolInfo]:
        """Return the tools advertised by the initialized MCP server."""
        if self._session is None:
            msg = "MCP session is not started"
            raise MCPProtocolError(msg)

        result = await self._session.list_tools()
        return [
            MCPToolInfo(
                name=tool.name,
                description=tool.description,
                input_schema=dict(tool.inputSchema),
            )
            for tool in result.tools
        ]
