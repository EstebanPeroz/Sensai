from typing import TYPE_CHECKING, override

from sensai_server.commands._base import Command

if TYPE_CHECKING:
    from sensai_server.commands import CommandRegistry
    from sensai_server.session import Session


class Help(Command):
    """Help command class to display available commands."""

    name = "help"
    help = "Display available commands."
    _registry: CommandRegistry

    def __init__(self, registry: CommandRegistry) -> None:
        """Initialize the Help command with a command registry."""
        super().__init__()
        self._registry = registry

    @override
    async def execute(self, session: Session, *args: str) -> None:
        """Display the available commands."""
        del args
        lines = ["Available commands:"]
        lines.extend(
            f"- {command.name}: {command.help}" for command in sorted(self._registry.commands(), key=lambda c: c.name)
        )
        await session.out.send_system_message("\n".join(lines))
