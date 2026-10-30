from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

import anyio
from anyio.from_thread import start_blocking_portal

from sensai.mcp.errors import MCPError
from sensai.mcp.session import MCPServerSession
from sensai.mcp.tool import MCPTool

if TYPE_CHECKING:
    from concurrent.futures import Future

    from anyio.abc import TaskStatus
    from anyio.from_thread import BlockingPortal

    from sensai.mcp.config import MCPServerConfig


class MCPHost:
    """Keep a session open to every enabled MCP server, on one background event loop, until `close`.

    Each session is opened and closed by its own long-lived task, as anyio requires, and tool calls made from
    other threads reach it through the blocking portal of that loop, so that a call does not restart its server.
    """

    def __init__(self, configs: list[MCPServerConfig]) -> None:
        """Store the server configurations without starting them."""
        self._configs = [config for config in configs if config.enabled]
        self._exit_stack = ExitStack()
        self._portal: BlockingPortal | None = None
        self._stop: anyio.Event | None = None
        self._serving: Future[None] | None = None

    def start(self) -> tuple[list[MCPTool], dict[str, MCPError]]:
        """Start every server concurrently, skipping the ones that fail.

        Returns the tools of the started servers, in the order of the configurations, and the error of each
        failed server by name, so that one unavailable server does not take the others down with it.
        """
        if not self._configs or self._portal is not None:
            return [], {}

        portal = self._exit_stack.enter_context(start_blocking_portal())
        self._portal = portal
        self._stop = portal.call(anyio.Event)
        sessions = [MCPServerSession(config) for config in self._configs]
        self._serving, started = portal.start_task(self._serve, portal, self._stop, sessions)
        return started

    def close(self) -> None:
        """Close every session, then stop the background event loop."""
        if self._portal is None or self._stop is None or self._serving is None:
            return

        try:
            self._portal.call(self._stop.set)
            self._serving.result()
        finally:
            self._portal = None
            self._exit_stack.close()

    @staticmethod
    async def _serve(
        portal: BlockingPortal,
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
                    infos = await serving.start(session.serve, portal, stop)
                except MCPError as err:
                    failures[session.server_name] = err
                else:
                    tools[index] = [MCPTool(info, session) for info in infos]

            async with anyio.create_task_group() as opening:
                for index, session in enumerate(sessions):
                    opening.start_soon(open_session, index, session)
            task_status.started(([tool for session_tools in tools for tool in session_tools], failures))
