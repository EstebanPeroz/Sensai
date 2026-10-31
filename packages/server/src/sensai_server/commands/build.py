from __future__ import annotations

from typing import TYPE_CHECKING

from sensai_server.commands._branch import Branch
from sensai_server.commands._effort import Effort
from sensai_server.commands._export import Export
from sensai_server.commands._help import Help
from sensai_server.commands._import import Import
from sensai_server.commands._model import Model
from sensai_server.commands._persona import PersonaCommand
from sensai_server.commands._tools import ToolsCommand

if TYPE_CHECKING:
    from sensai_server.commands import CommandRegistry
    from sensai_server.commands._base import Command
    from sensai_server.llm.provider import ProviderManager
    from sensai_server.memory.persistent_db.adapter import PersistentDatabase
    from sensai_server.session import SessionRegistry
    from sensai_server.tools.tool_manager import ToolManager


def build_commands(
    sessions: SessionRegistry,
    commands_registry: CommandRegistry,
    db: PersistentDatabase,
    pm: ProviderManager,
    tool_manager: ToolManager,
) -> list[Command]:
    """Build every command of the server, to register in the `CommandRegistry`.

    `sessions` are the live sessions, for the commands that act on other sessions than the caller's. `tool_manager`
    is shared by every session, each one toggling its tools in its own state.
    """
    return [
        Model(),
        PersonaCommand(sessions),
        Branch(db, pm),
        ToolsCommand(tool_manager),
        Effort(),
        Export(db),
        Import(db),
        Help(commands_registry),
    ]
