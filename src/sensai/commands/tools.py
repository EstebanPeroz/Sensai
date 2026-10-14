from typing import TYPE_CHECKING, override

from sensai.commands.base import Command

if TYPE_CHECKING:
    from sensai.tools.tool_manager import ToolManager
    from sensai.ui.adapter import UIAdapter

ERROR_LEN_CMD = 2


class ToolsCommand(Command):
    """Command class to list available tools and toggle their availability."""

    name = "tools"
    help = "List available tools and toggle them."
    usage = "/tools [list|enable|disable] [name]"
    _tool_manager: ToolManager

    def __init__(self, ui: UIAdapter, tool_manager: ToolManager) -> None:
        """Initialize the command with the tool registry it manages."""
        super().__init__(ui)
        self._tool_manager = tool_manager

    @override
    def execute(self, *args: str) -> None:
        """List the tools or toggle one by name."""
        if not args or args[0] == "list":
            self._list_tools()
            return
        if args[0] == "enable":
            if len(args) != ERROR_LEN_CMD:
                self._ui.send_system_message(f"Usage: {self.usage}")
                return
            self._enable_tool(args[1])
            return
        if args[0] == "disable":
            if len(args) != ERROR_LEN_CMD:
                self._ui.send_system_message(f"Usage: {self.usage}")
                return
            self._disable_tool(args[1])
            return

        self._ui.send_system_message(f"Usage: {self.usage}")

    @override
    def completions(self) -> list[str]:
        """Return the supported subcommands and tool names."""
        return ["list", "enable", "disable", *self._tool_manager.names()]

    def _list_tools(self) -> None:
        """Display every registered tool and whether it is currently enabled."""
        names = self._tool_manager.names()
        if not names:
            self._ui.send_system_message("No tools are available.")
            return

        lines = ["Available tools:"]
        lines.extend(f"- {name}: {'enabled' if self._tool_manager.is_enabled(name) else 'disabled'}" for name in names)
        self._ui.send_system_message("\n".join(lines))

    def _enable_tool(self, name: str) -> None:
        """Enable a disabled tool if it exists."""
        if self._tool_manager.enable(name):
            self._ui.send_system_message(f"Tool '{name}' enabled.", append_response=True)
            return
        self._ui.send_system_message(f"Tool '{name}' is not available.")

    def _disable_tool(self, name: str) -> None:
        """Disable an enabled tool if it exists."""
        if self._tool_manager.disable(name):
            self._ui.send_system_message(f"Tool '{name}' disabled.", append_response=True)
            return
        self._ui.send_system_message(f"Tool '{name}' is not available.")
