from __future__ import annotations

from typing import TYPE_CHECKING

from sensai_server.commands.persona import PersonaCommand

if TYPE_CHECKING:
    from sensai_server.commands.base import Command
    from sensai_server.session import SessionRegistry


def build_commands(sessions: SessionRegistry) -> list[Command]:
    """Build every command of the server, to register in the `CommandRegistry`.

    `sessions` are the live sessions, for the commands that act on other sessions than the caller's.
    """
    return [PersonaCommand(sessions)]
