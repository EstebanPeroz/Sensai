from typing import TYPE_CHECKING, override

from sensai.commands.base import Command

if TYPE_CHECKING:
    from collections.abc import Callable

    from sensai.tools.tool_manager import ToolManager

ERROR_LEN_CMD = 2


class ToolsCommand(Command):
    """Command class to list available tools and toggle their availability."""

    name = "tools"
    help = "List available tools and toggle them."
    usage = "/tools [list|enable|disable] [name]"
    _tool_manager: ToolManager
    _subcommands: dict[str, Callable[..., None]]

    def __init__(self, tool_manager: ToolManager) -> None:
        """Initialize the command with the tool registry it manages."""
        super().__init__()
        self._subcommands = {
            "list": self._list_tools,
            "enable": self._enable_tool,
            "disable": self._disable_tool,
        }
        self._tool_manager = tool_manager

    @override
    def execute(self, *args: str) -> None:
        """List the tools or toggle the one named, dispatching on `_subcommands`."""
        if not args or args[0] == "list":
            self._list_tools()
            return

        subcommand = self._subcommands.get(args[0])
        if subcommand is None or len(args) != ERROR_LEN_CMD:
            self._ui.send_system_message(f"Usage: {self.usage}")
            return
        subcommand(args[1])

    def _list_tools(self) -> None:
        """Display every registered tool and whether it is currently enabled."""
        names = self._tool_manager.names()
        if not names:
            self._ui.send_system_message("No tools are available.")
            return

        lines = ["Available tools:"]
        lines.extend(f"- {name}: {self._status(name)}" for name in names)
        self._ui.send_system_message("\n".join(lines))

    def _status(self, name: str) -> str:
        """Describe whether the tool `name` is enabled, and whether it asks for confirmation."""
        status = "enabled" if self._tool_manager.is_enabled(name) else "disabled"
        if self._tool_manager.needs_confirmation(name):
            status += ", asks confirmation"
        return status

    def _enable_tool(self, name: str) -> None:
        """Enable a disabled tool if it exists."""
        if self._tool_manager.enable(name):
            self._ui.send_system_message(f"Tool '{name}' enabled.", append_response=True)
            return
        return

    def _disable_tool(self, name: str) -> None:
        """Disable an enabled tool if it exists."""
        if self._tool_manager.disable(name):
            self._ui.send_system_message(f"Tool '{name}' disabled.", append_response=True)
            return
        self._ui.send_system_message(f"Tool '{name}' is not available.")

    @override
    def completions(self, args: list[str]) -> list[str]:
        """Return content for the completion with tool command."""
        size: int = len(args)
        if size == 0:
            return sorted(self._subcommands)
        if size == 1 and args[0] in ["disable", "enable"]:
            return self._tool_manager.names()
        return []
