from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from sensai.mcp.discovery import MCPToolInfo
from sensai.mcp.errors import MCPCallError, MCPProtocolError, MCPShutdownError, MCPStartupError

if TYPE_CHECKING:
    from sensai.mcp.config import MCPServerConfig


class MCPServerSession:
    """Manage one initialized MCP server session over stdio."""

    def __init__(self, config: MCPServerConfig) -> None:
        """Store the server configuration without starting the process."""
        self._config = config
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    @property
    def server_name(self) -> str:
        """Return the configured name of the server."""
        return self._config.name

    async def start(self) -> None:
        """Start the configured server and initialize its MCP session."""
        if self._session is not None:
            return

        exit_stack = AsyncExitStack()
        try:
            streams = await exit_stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=self._config.command,
                        args=self._config.args,
                    ),
                )
            )
            session = await exit_stack.enter_async_context(ClientSession(*streams))
            with anyio.fail_after(self._config.startup_timeout):
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
                input_schema=dict(tool.input_schema),
            )
            for tool in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, object]) -> str:
        """Call one MCP tool and return its simple text result."""
        if self._session is None:
            msg = "MCP session is not started"
            raise MCPCallError(msg)

        try:
            with anyio.fail_after(self._config.call_timeout):
                result = await self._session.call_tool(name, arguments)
        except Exception as err:
            if isinstance(err, MCPCallError):
                raise
            raise MCPCallError(str(err)) from err

        is_error = getattr(result, "is_error", None)
        if is_error is None:
            is_error = getattr(result, "isError", False)
        if is_error:
            msg = "MCP tool returned an error"
            raise MCPCallError(msg)
        if len(result.content) != 1 or result.content[0].type != "text":
            msg = "MCP tool returned a non-text result"
            raise MCPCallError(msg)
        return result.content[0].text

    def call_tool_sync(self, name: str, arguments: dict[str, object]) -> str:
        """Call one MCP tool synchronously in a self-contained MCP lifecycle."""
        with anyio.from_thread.start_blocking_portal() as portal:
            return portal.call(self._call_tool_in_new_session, name, arguments)

    async def _call_tool_in_new_session(self, name: str, arguments: dict[str, object]) -> str:
        """Start, call, and close a temporary session in one event loop."""
        session = MCPServerSession(self._config)
        await session.start()
        try:
            return await session.call_tool(name, arguments)
        finally:
            await session.close()
