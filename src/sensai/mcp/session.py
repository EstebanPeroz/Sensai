from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

import anyio
import httpx2
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from sensai.mcp.discovery import MCPToolInfo
from sensai.mcp.errors import MCPCallError, MCPProtocolError, MCPShutdownError, MCPStartupError, describe

if TYPE_CHECKING:
    from anyio.abc import TaskStatus
    from anyio.from_thread import BlockingPortal

    from sensai.mcp.config import MCPServerConfig

logger = logging.getLogger(__name__)

# The MCP SDK's own defaults: a server may hold a response stream open, hence the long read timeout.
# The startup and call timeouts of the configuration still bound every request made through the session.
_HTTP_TIMEOUT = httpx2.Timeout(30.0, read=300.0)


class MCPServerSession:
    """Manage one initialized MCP server session, over stdio or streamable HTTP."""

    def __init__(self, config: MCPServerConfig) -> None:
        """Store the server configuration without starting the process."""
        self._config = config
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._portal: BlockingPortal | None = None
        self._lost: str | None = None

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
            session = await self._open_session(exit_stack)
            with anyio.fail_after(self._config.startup_timeout):
                await session.initialize()
        except BaseException as err:
            # A transport whose background task fails, such as an http connection refused, cancels `start` and
            # only raises the failure itself once closed: always close it, and report that failure instead.
            cause = err
            try:
                await exit_stack.aclose()
            except Exception as close_err:  # noqa: BLE001
                cause = close_err
            if not isinstance(cause, Exception):
                raise
            raise MCPStartupError.from_error(cause, self._config.startup_timeout) from cause

        self._exit_stack = exit_stack
        self._session = session

    async def _open_session(self, exit_stack: AsyncExitStack) -> ClientSession:
        """Connect to the server over its transport, leaving `exit_stack` to disconnect it.

        A validated configuration has a url for the http transport and a command for the stdio one.
        """
        if self._config.url is not None:
            http_client = await exit_stack.enter_async_context(
                httpx2.AsyncClient(headers=self._config.headers, timeout=_HTTP_TIMEOUT),
            )
            streams = await exit_stack.enter_async_context(
                streamable_http_client(self._config.url, http_client=http_client),
            )
        elif self._config.command is not None:
            streams = await exit_stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=self._config.command,
                        args=self._config.args,
                    ),
                )
            )
        else:
            msg = "MCP server has neither a url nor a command"
            raise MCPStartupError(msg)
        return await exit_stack.enter_async_context(ClientSession(*streams))

    async def close(self) -> None:
        """Close the MCP session and its server process."""
        if self._exit_stack is None:
            return

        try:
            await self._exit_stack.aclose()
        except Exception as err:
            raise MCPShutdownError(describe(err)) from err
        finally:
            self._exit_stack = None
            self._session = None

    async def serve(
        self,
        portal: BlockingPortal,
        stop: anyio.Event,
        *,
        task_status: TaskStatus[list[MCPToolInfo]] = anyio.TASK_STATUS_IGNORED,
    ) -> None:
        """Open the session, report its tools, then keep it open for `call_tool_sync` until `stop` is set.

        The session is opened and closed by this one task, as anyio requires of the task group of its transport,
        while calls reach it from other threads through `portal`, the portal of the event loop running this task.

        Raises:
            MCPStartupError: Before reporting the tools, if the server cannot start or list them in time.

        """
        await self.start()
        try:
            with anyio.fail_after(self._config.startup_timeout):
                tools = await self.list_tools()
        except BaseException as err:
            cause = await self._close_after(err)
            if not isinstance(cause, Exception):
                raise
            raise MCPStartupError.from_error(cause, self._config.startup_timeout) from cause

        self._portal = portal
        task_status.started(tools)
        try:
            await stop.wait()
        except BaseException as err:  # noqa: BLE001
            # A transport whose connection drops cancels the task that opened it: the session is gone for good.
            self._lost = describe(await self._close_after(err))
            logger.warning("MCP server '%s' disconnected: %s", self.server_name, self._lost)
            return
        finally:
            self._portal = None
        try:
            await self.close()
        except MCPShutdownError as err:
            logger.warning("MCP server '%s' did not close cleanly: %s", self.server_name, err)

    async def _close_after(self, err: BaseException) -> BaseException:
        """Close the session after `err`, and return the failure of its transport if closing reveals one.

        A transport whose background task fails, such as a dropped connection, cancels the task that opened it
        and only raises the failure itself once closed. Not shielded: closing exits the cancel scopes of the
        transport, which anyio refuses from inside a scope entered after them.
        """
        try:
            await self.close()
        except MCPShutdownError as close_err:
            return close_err.__cause__ or close_err
        return err

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
        except TimeoutError as err:
            msg = f"no answer within {self._config.call_timeout:g}s"
            raise MCPCallError(msg) from err
        except Exception as err:
            if isinstance(err, MCPCallError):
                raise
            raise MCPCallError(describe(err)) from err

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
        """Call one MCP tool from a thread other than the event loop of the session, kept open by `serve`."""
        if self._lost is not None:
            msg = f"MCP server '{self.server_name}' disconnected, restart Sensai to reconnect: {self._lost}"
            raise MCPCallError(msg)
        if self._portal is None:
            msg = "MCP session is not started"
            raise MCPCallError(msg)
        return self._portal.call(self.call_tool, name, arguments)
