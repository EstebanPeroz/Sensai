from __future__ import annotations

from typing import TYPE_CHECKING

from sensai_protocol.error import SensaiError

if TYPE_CHECKING:
    from sensai_server.commands._base import Command
    from sensai_server.session import Session


class UnknownCommandError(SensaiError):
    """Exception raised when an unknown command is encountered."""

    def __init__(self, command_name: str) -> None:
        """Initialize the exception with the unknown command name."""
        self.command_name = command_name
        super().__init__(f"Unknown command: {command_name}")


class CommandRegistry:
    """The commands of the server, by name, shared by every session."""

    _commands: dict[str, Command]

    def __init__(self) -> None:
        """Initialize the command registry."""

    def set_commands(self, commands: list[Command]) -> None:
        """Set the commands in the registry."""
        self._commands = {command.name: command for command in commands}

    def names(self) -> list[str]:
        """Return the typed names of the commands, such as `/persona`, for the client's completion."""
        return [f"/{name}" for name in self._commands]

    def commands(self) -> list[Command]:
        """Return a list of registered commands."""
        return list(self._commands.values())

    async def execute(self, session: Session, line: str) -> None:
        """Parse `line` as a `/name args...` command and run it for `session`. An unknown name is reported."""
        parts = line.removeprefix("/").split()
        if not parts:
            return
        name, *args = parts
        command = self._commands.get(name)
        if command is None:
            await session.out.send_system_message(str(UnknownCommandError(name)))
            return
        await command.execute(session, *args)

    def completions(self, session: Session, text: str) -> list[str]:
        """Return the values of the word being typed in the input line `text`, after its finished words."""
        words = text.split()
        if not text.startswith("/") or not words:
            return []
        name, *args = words
        if not args and not text[-1].isspace():
            return []
        finished = args if text[-1].isspace() else args[:-1]
        command = self._commands.get(name.removeprefix("/"))
        if command is None:
            return []
        return command.completions(session, finished)
