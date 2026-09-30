from typing import override

from sensai.commands.base import Command


class Help(Command):
    """Help command class to display available commands."""

    name = "help"
    help = "Display available commands."

    @classmethod
    @override
    def execute(cls, *args: str) -> None:
        """Print the available commands."""
        print("Available commands:")
        for cls_instance in Command.__subclasses__():
            help_text = cls_instance.help
            print(f"- {cls_instance.__name__.lower()}: {help_text}")
