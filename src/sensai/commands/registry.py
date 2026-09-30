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

    _commands: dict[str, Command]

    def __init__(self, commands: list[Command]) -> None:
        """Initialize the command registry."""
        self._commands = {command.name: command for command in commands}

    def register_command(self, command: Command) -> None:
        """Register a command with its associated function."""
        self._commands[command.name] = command

    def execute_command(self, command_name: str, *args: str) -> None:
        """Execute a registered command with the given arguments."""
        if command_name not in self._commands:
            raise UnknownCommandError(command_name)
        self._commands[command_name].execute(*args)

    def commands(self) -> list[Command]:
        """Return a list of registered commands."""
        return list(self._commands.values())
