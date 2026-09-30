from typing import TYPE_CHECKING, override

from sensai.commands.base import Command

if TYPE_CHECKING:
    from sensai.commands.registry import CommandRegistry


class Help(Command):
    """Help command class to display available commands."""

    name = "help"
    help = "Display available commands."
    _registry: CommandRegistry

    def __init__(self, registry: CommandRegistry) -> None:
        """Initialize the Help command with a command registry."""
        self._registry = registry

    @override
    def execute(self, *args: str) -> None:
        """Print the available commands."""
        print("Available commands:")
        for command in self._registry.commands():
            print(f"- {command.name}: {command.help}")
