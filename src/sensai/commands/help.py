from typing import overload

from sensai.commands.base import Command


class Help(Command):
    """Help command class to display available commands."""

    @overload
    def execute(self, *args: str) -> None:
        print("Available commands:")
