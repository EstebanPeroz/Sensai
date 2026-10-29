from __future__ import annotations

from typing import TYPE_CHECKING

from sensai_server.commands._branch import Branch
from sensai_server.commands._help import Help
from sensai_server.commands._persona import PersonaCommand

if TYPE_CHECKING:
    from sensai_server.commands import CommandRegistry
    from sensai_server.commands._base import Command
    from sensai_server.llm.provider import ProviderManager
    from sensai_server.memory.persistent_db.adapter import PersistentDatabase
    from sensai_server.session import SessionRegistry


def build_commands(
    sessions: SessionRegistry, commands: CommandRegistry, db: PersistentDatabase, pm: ProviderManager
) -> list[Command]:
    """Build every command of the server, to register in the `CommandRegistry`.

    `sessions` are the live sessions, for the commands that act on other sessions than the caller's.
    """
    return [PersonaCommand(sessions), Help(commands), Branch(db, pm)]
