from __future__ import annotations

from typing import TYPE_CHECKING, override

from sensai_server.commands._base import Command

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sensai_server.session import Session
    from sensai_server.tools.state import ToolState
    from sensai_server.tools.tool_manager import ToolManager

ERROR_LEN_CMD = 2


class ToolsCommand(Command):
    """Command class to list the tools of a session and toggle their availability, in that session only."""

    name = "tools"
    help = "List available tools and toggle them."
    usage = "/tools [list|enable|disable] [name]"
    _tool_manager: ToolManager
    _subcommands: dict[str, Callable[[Session, str], Awaitable[None]]]

    def __init__(self, tool_manager: ToolManager) -> None:
        """Initialize the command with the tool registry it manages, shared by every session."""
        self._tool_manager = tool_manager
        self._subcommands = {
            "list": self._list_subcommand,
            "enable": self._enable_tool,
            "disable": self._disable_tool,
        }

    @override
    async def execute(self, session: Session, *args: str) -> None:
        """List the tools of `session` or toggle the one named, dispatching on `_subcommands`."""
        if not args or args[0] == "list":
            await self._list_tools(session)
            return

        subcommand = self._subcommands.get(args[0])
        if subcommand is None or len(args) != ERROR_LEN_CMD:
            await session.out.send_system_message(f"Usage: {self.usage}")
            return
        await subcommand(session, args[1])

    async def _list_subcommand(self, session: Session, _name: str) -> None:
        """Run `list` as a subcommand: its argument is ignored."""
        await self._list_tools(session)

    async def _list_tools(self, session: Session) -> None:
        """Display every registered tool and whether it is currently enabled in `session`."""
        names = self._tool_manager.names()
        if not names:
            await session.out.send_system_message("No tools are available.")
            return

        lines = ["Available tools:"]
        lines.extend(f"- {name}: {self._status(session.tools, name)}" for name in names)
        await session.out.send_system_message("\n".join(lines))

    def _status(self, state: ToolState, name: str) -> str:
        """Describe whether the tool `name` is enabled in `state`, and whether it asks for confirmation."""
        status = "enabled" if self._tool_manager.is_enabled(state, name) else "disabled"
        if self._tool_manager.needs_confirmation(state, name):
            status += ", asks confirmation"
        return status

    async def _enable_tool(self, session: Session, name: str) -> None:
        """Enable a tool in `session` if it exists."""
        if self._tool_manager.enable(session.tools, name):
            await session.out.send_system_message(f"Tool '{name}' enabled.", append_response=True)
            return
        await session.out.send_system_message(f"Tool '{name}' is not available.")

    async def _disable_tool(self, session: Session, name: str) -> None:
        """Disable a tool in `session` if it exists."""
        if self._tool_manager.disable(session.tools, name):
            await session.out.send_system_message(f"Tool '{name}' disabled.", append_response=True)
            return
        await session.out.send_system_message(f"Tool '{name}' is not available.")

    @override
    def completions(self, session: Session, args: list[str]) -> list[str]:
        """Return the subcommands, then the tool names after `enable` or `disable`."""
        del session
        size: int = len(args)
        if size == 0:
            return sorted(self._subcommands)
        if size == 1 and args[0] in ["disable", "enable"]:
            return self._tool_manager.names()
        return []
