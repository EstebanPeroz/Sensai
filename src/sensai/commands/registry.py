from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sensai.commands.base import Command


class UnknownCommandError(Exception):
    """Exception raised when an unknown command is encountered."""

    def __init__(self, command_name: str) -> None:
        """Initialize the exception with the unknown command name."""
        self.command_name = command_name
        super().__init__(f"Unknown command: {command_name}")


class CommandRegistry:
    """Contain all commands and their associated functions."""

    commands: dict[str, Command]

    def __init__(self) -> None:
        """Initialize the command registry."""
        self.commands = {}

    def register_command(self, command: Command) -> None:
        """Register a command with its associated function."""
        self.commands[command.name] = command

    def execute_command(self, command_name: str, *args: str) -> None:
        """Execute a registered command with the given arguments."""
        if command_name not in self.commands:
            raise UnknownCommandError(command_name)
        command = self.commands[command_name]
        command.execute(*args)
