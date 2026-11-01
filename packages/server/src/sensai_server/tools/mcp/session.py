from __future__ import annotations

import logging
from contextlib import AsyncExitStack, suppress
from typing import TYPE_CHECKING

import anyio
import httpx2
from mcp import ClientSession, StdioServerParameters
from mcp import MCPError as SDKError
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CONNECTION_CLOSED, CallToolResult

from sensai_server.tools.mcp.discovery import MCPToolInfo
from sensai_server.tools.mcp.errors import (
    MCPCallError,
    MCPProtocolError,
    MCPShutdownError,
    MCPStartupError,
    describe,
)

if TYPE_CHECKING:
    from anyio.abc import TaskStatus

    from sensai_server.config.settings import MCPServerConfig

logger = logging.getLogger(__name__)

# The MCP SDK's own defaults: a server may hold a response stream open, hence the long read timeout.
# The startup and call timeouts of the configuration still bound every request made through the session.
_HTTP_TIMEOUT = httpx2.Timeout(30.0, read=300.0)


class MCPServerSession:
    """Manage one initialized MCP server session, over stdio or streamable HTTP.

    Fully async: it only ever runs on the event loop of the MCP host, which opens it, calls it and closes it.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        """Store the server configuration without starting the process."""
        self._config = config
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._lost: str | None = None
        self._dropped = anyio.Event()

    @property
    def server_name(self) -> str:
        """Return the configured name of the server."""
        return self._config.name

    async def start(self) -> None:
        """Start the configured server and initialize its MCP session.

        Raises:
            MCPStartupError: If the server cannot start or initialize within `startup_timeout`.

        """
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
                stdio_client(StdioServerParameters(command=self._config.command, args=self._config.args)),
            )
        else:
            msg = "MCP server has neither a url nor a command"
            raise MCPStartupError(msg)
        return await exit_stack.enter_async_context(ClientSession(*streams))

    async def close(self) -> None:
        """Close the MCP session and its server process.

        Raises:
            MCPShutdownError: If the transport fails while closing.

        """
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
        stop: anyio.Event,
        *,
        task_status: TaskStatus[list[MCPToolInfo]] = anyio.TASK_STATUS_IGNORED,
    ) -> None:
        """Open the session, report its tools, then keep it open for `call_tool` until `stop` is set.

        The session is opened and closed by this one task, as anyio requires of the task group of its transport.
        A dropped connection marks the session lost: every later call fails, and it is never reopened. An http
        transport cancels this task when its connection drops. A stdio one does not: a dead subprocess only shows
        as a closed connection on the next call, which wakes this task through `_dropped`.

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

        task_status.started(tools)
        try:
            await _wait_any(stop, self._dropped)
        except BaseException as err:  # noqa: BLE001
            # A transport whose connection drops cancels the task that opened it: the session is gone for good.
            self._lost = describe(await self._close_after(err))
            logger.warning("MCP server '%s' disconnected: %s", self.server_name, self._lost)
            return
        if self._dropped.is_set():
            with suppress(MCPShutdownError):
                await self.close()
            logger.warning("MCP server '%s' disconnected: %s", self.server_name, self._lost)
            return
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
        """Return the tools advertised by the initialized MCP server.

        Raises:
            MCPProtocolError: If the session is not started.

        """
        if self._session is None:
            msg = "MCP session is not started"
            raise MCPProtocolError(msg)

        result = await self._session.list_tools()
        return [
            MCPToolInfo(name=tool.name, description=tool.description, input_schema=dict(tool.input_schema))
            for tool in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, object]) -> str:
        """Call one MCP tool and return its single text result.

        Raises:
            MCPCallError: If the server disconnected, the session is not started, the call fails or times out,
                or the result is an error or not exactly one text block.

        """
        if self._lost is not None:
            raise self._lost_error()
        if self._session is None:
            msg = "MCP session is not started"
            raise MCPCallError(msg)

        try:
            with anyio.fail_after(self._config.call_timeout):
                result = await self._session.call_tool(name, arguments)
        except TimeoutError as err:
            msg = f"no answer within {self._config.call_timeout:g}s"
            raise MCPCallError(msg) from err
        except SDKError as err:
            if err.code != CONNECTION_CLOSED:
                raise MCPCallError(describe(err)) from err
            # The transport is gone, such as a dead stdio subprocess: `serve` closes the session.
            if self._lost is None:
                self._lost = describe(err)
            self._dropped.set()
            raise self._lost_error() from err
        except Exception as err:
            raise MCPCallError(describe(err)) from err
        return _text_of(result)

    def _lost_error(self) -> MCPCallError:
        """Return the error of a call to this server once its connection dropped."""
        msg = f"MCP server '{self.server_name}' disconnected, restart the server to reconnect: {self._lost}"
        return MCPCallError(msg)


def _text_of(result: CallToolResult) -> str:
    """Return the single text block of a tool result.

    Raises:
        MCPCallError: If the result is an error, or not exactly one text block.

    """
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


async def _wait_any(*events: anyio.Event) -> None:
    """Return as soon as one of `events` is set."""
    async with anyio.create_task_group() as task_group:

        async def wait(event: anyio.Event) -> None:
            await event.wait()
            task_group.cancel_scope.cancel()

        for event in events:
            task_group.start_soon(wait, event)
