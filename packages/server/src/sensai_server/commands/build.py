from __future__ import annotations

from typing import TYPE_CHECKING

from sensai_server.commands._branch import Branch
from sensai_server.commands._dashboard import Dashboard
from sensai_server.commands._effort import Effort
from sensai_server.commands._export import Export
from sensai_server.commands._help import Help
from sensai_server.commands._import import Import
from sensai_server.commands._index import Index
from sensai_server.commands._model import Model
from sensai_server.commands._persona import PersonaCommand
from sensai_server.commands._tools import ToolsCommand

if TYPE_CHECKING:
    from sensai_server.commands import CommandRegistry
    from sensai_server.commands._base import Command
    from sensai_server.llm.provider import ProviderManager
    from sensai_server.memory.persistent_db.adapter import PersistentDatabase
    from sensai_server.monitoring.budget import ContextBudget
    from sensai_server.rag import Rag
    from sensai_server.session import SessionRegistry
    from sensai_server.tools.tool_manager import ToolManager


def build_commands(  # noqa: PLR0913 - every command's shared dependency comes through here
    sessions: SessionRegistry,
    commands_registry: CommandRegistry,
    db: PersistentDatabase,
    pm: ProviderManager,
    tool_manager: ToolManager,
    *,
    rag: Rag | None = None,
    budget: ContextBudget | None = None,
) -> list[Command]:
    """Build every command of the server, to register in the `CommandRegistry`.

    `sessions` are the live sessions, for the commands that act on other sessions than the caller's. `tool_manager`
    is shared by every session, each one toggling its tools in its own state. `/index` is left out without `rag`. The
    dashboard reads `db.interactions` and is always present: with logging off, its report is just empty. `/model`
    and `/branch` send the new context usage from `budget`, if any.
    """
    commands: list[Command] = [
        Model(budget),
        PersonaCommand(sessions),
        Branch(db, pm, budget),
        ToolsCommand(tool_manager),
        Effort(),
        Export(db),
        Import(db),
        Dashboard(db.interactions),
    ]
    if rag is not None:
        commands.append(Index(rag))
    commands.append(Help(commands_registry))
    return commands
