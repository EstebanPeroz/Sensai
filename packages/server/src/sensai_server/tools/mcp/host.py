from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

import anyio
from anyio.from_thread import start_blocking_portal

from sensai_server.tools.mcp.errors import MCPCallError, MCPError
from sensai_server.tools.mcp.session import MCPServerSession
from sensai_server.tools.mcp.tool import MCPTool

if TYPE_CHECKING:
    from concurrent.futures import Future

    from anyio.abc import TaskStatus
    from anyio.from_thread import BlockingPortal

    from sensai_server.config.settings import MCPServerConfig


class MCPHost:
    """Keep a session open to every enabled MCP server, on its own event loop in a background thread, until `close`.

    The sessions don't run on uvicorn's loop: the host starts before uvicorn does, and anyio requires each
    transport to be opened and closed by one long-lived task. Each session is served by its own task on the
    host's loop, and a tool call made from uvicorn's loop reaches it through `call`, so that no call starts or
    stops a server. The portal to that loop never leaves the host.
    """

    def __init__(self, configs: list[MCPServerConfig]) -> None:
        """Keep the enabled server configurations, without starting anything."""
        self._configs = [config for config in configs if config.enabled]
        self._exit_stack = ExitStack()
        self._portal: BlockingPortal | None = None
        self._stop: anyio.Event | None = None
        self._serving: Future[None] | None = None
        self._closed = False

    def start(self) -> tuple[list[MCPTool], dict[str, MCPError]]:
        """Start every enabled server concurrently, skipping the ones that fail.

        Returns the tools of the started servers, in the order of the configurations, and the error of each
        failed server by name, so that one unavailable server does not take the others down with it. With no
        enabled server, no thread is started.
        """
        if not self._configs or self._portal is not None or self._closed:
            return [], {}

        portal = self._exit_stack.enter_context(start_blocking_portal())
        self._portal = portal
        try:
            self._stop = portal.call(anyio.Event)
            sessions = [MCPServerSession(config) for config in self._configs]
            self._serving, started = portal.start_task(self._serve, self._stop, sessions)
        except BaseException:
            self._portal = None
            self._exit_stack.close()
            raise
        return started

    def call(self, session: MCPServerSession, name: str, arguments: dict[str, object]) -> Future[str]:
        """Call the tool `name` of `session` on the host's loop, returning the future of its text result.

        Cancelling the future cancels the call on the host's loop.

        Raises:
            MCPCallError: If the host is not started or already closed.

        """
        if self._portal is None:
            msg = "MCP host is closed"
            raise MCPCallError(msg)
        try:
            return self._portal.start_task_soon(session.call_tool, name, arguments)
        except RuntimeError as err:
            # `close` stopped the portal from another thread since the check above.
            msg = "MCP host is closed"
            raise MCPCallError(msg) from err

    def close(self) -> None:
        """Close every session, then stop the background thread and its loop. Idempotent.

        Blocks until every session is closed and every subprocess reaped.
        """
        self._closed = True
        portal, stop, serving = self._portal, self._stop, self._serving
        if portal is None:
            return

        self._portal = None
        try:
            if stop is not None and serving is not None:
                portal.call(stop.set)
                serving.result()
        finally:
            self._exit_stack.close()

    async def _serve(
        self,
        stop: anyio.Event,
        sessions: list[MCPServerSession],
        *,
        task_status: TaskStatus[tuple[list[MCPTool], dict[str, MCPError]]],
    ) -> None:
        """Open every session concurrently, report their tools, then keep them open until `stop` is set."""
        tools: list[list[MCPTool]] = [[] for _ in sessions]
        failures: dict[str, MCPError] = {}

        async with anyio.create_task_group() as serving:

            async def open_session(index: int, session: MCPServerSession) -> None:
                try:
                    infos = await serving.start(session.serve, stop)
                except MCPError as err:
                    failures[session.server_name] = err
                else:
                    tools[index] = [MCPTool(info, session, self) for info in infos]

            async with anyio.create_task_group() as opening:
                for index, session in enumerate(sessions):
                    opening.start_soon(open_session, index, session)
            # Failures by name, in the order of the configurations rather than the order they happened in.
            ordered = {s.server_name: failures[s.server_name] for s in sessions if s.server_name in failures}
            task_status.started(([tool for session_tools in tools for tool in session_tools], ordered))
