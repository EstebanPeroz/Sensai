from typing import TYPE_CHECKING, override

from sensai.commands.base import Command

if TYPE_CHECKING:
    from sensai.commands.registry import CommandRegistry
    from sensai.ui.adapter import UIAdapter


class Help(Command):
    """Help command class to display available commands."""

    name = "help"
    help = "Display available commands."
    _registry: CommandRegistry

    def __init__(self, ui: UIAdapter, registry: CommandRegistry) -> None:
        """Initialize the Help command with a command registry."""
        super().__init__(ui)
        self._registry = registry

    @override
    def execute(self, *args: str) -> None:
        """Display the available commands."""
        lines = ["Available commands:"]
        lines.extend(
            f"- {command.name}: {command.help}" for command in sorted(self._registry.commands(), key=lambda c: c.name)
        )
        self._ui.send_system_message("\n".join(lines))
