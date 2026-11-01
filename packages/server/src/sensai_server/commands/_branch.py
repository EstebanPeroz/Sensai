from typing import TYPE_CHECKING, override
from uuid import UUID

from sensai_protocol.server_messages import HistoryMessage
from sensai_protocol.server_messages import Role as HistoryRole

import sensai_server.memory.persistent_db.error as db_err
from sensai_server.cache import CacheError
from sensai_server.chat.usage import send_context_usage
from sensai_server.commands._base import Command

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sensai_server.llm.provider import ProviderManager
    from sensai_server.memory.persistent_db.adapter import PersistentDatabase
    from sensai_server.monitoring.budget import ContextBudget
    from sensai_server.session import Session


class Branch(Command):
    """Branch command class to display or choose available models."""

    name = "branch"
    help = "Update the branch or load them"
    usage = "/branch list, branch set <uuid>, branch remove <uuid>, branch rename <uuid> <new_name>, branch fork"

    _subcommands: dict[str, Callable[..., Awaitable[None]]]
    _db: PersistentDatabase
    _pm: ProviderManager
    _budget: ContextBudget | None

    def __init__(self, db: PersistentDatabase, pm: ProviderManager, budget: ContextBudget | None = None) -> None:
        """Initialize the Branch command and its subcommand map.

        A loaded branch is sent its context usage from `budget`, if any.

        Raise:
            DatabaseError: Failed to get db

        """
        super().__init__()
        self._subcommands = {
            "list": self._list,
            "set": self._set,
            "remove": self._remove,
            "rename": self._rename,
            "fork": self._fork,
        }
        self._db = db
        self._pm = pm
        self._budget = budget

    async def _list(self, session: Session) -> None:
        """Display the available branches."""
        branches: dict[UUID, str] = self._db.get_branches()
        for i, (branch_id, name) in enumerate(branches.items()):
            await session.out.send_system_message(
                str(branch_id) + ": " + name,
                append_response=(i != 0),
            )

    async def _set(self, session: Session, uuid: str) -> None:
        """Load the branch identified by uuid.

        Switching to another branch clears the session's semantic cache, so no answer cached in the previous
        conversation is reused. A failed switch, or one to the current branch, keeps it. The loaded history is
        followed by the branch's context usage.
        """
        target_id: UUID | None = self._try_parse_uuid(uuid)
        if target_id is None:
            await session.out.send_system_message("parameter is not a valid UUID.")
            return
        previous_id: UUID | None = session.conv.uuid
        try:
            self._db.get_branch(
                current=session.conv,
                target_branch=target_id,
                provider_manager=self._pm,
            )
        except db_err.InvalidInstanceError as err:
            await session.out.send_system_message(str(err))
            return
        if session.conv.uuid != previous_id:
            await self._clear_cache(session)
        messages: list[HistoryMessage] = [
            HistoryMessage(uuid=message.uuid, role=HistoryRole(message.role.value), content=message.content)
            for message in session.conv.history.messages()
        ]
        await session.out.load_conversation(messages)
        await send_context_usage(session, self._budget)

    async def _remove(self, session: Session, uuid: str) -> None:
        """Remove the branch identified by uuid."""
        target_id: UUID | None = self._try_parse_uuid(uuid)
        if target_id is None:
            await session.out.send_system_message("parameter is not a valid UUID.")
            return
        if target_id == session.conv.uuid:
            await session.out.send_system_message("You cannot remove your current branch.")
            return
        try:
            self._db.remove_branch(target_id)
        except db_err.InvalidInstanceError:
            await session.out.send_system_message(f"Failed to remove {uuid}")
            return
        await session.out.send_system_message(f"Branch {uuid} removed")

    async def _rename(self, session: Session, uuid: str, new_name: str) -> None:
        """Rename the branch identified by uuid."""
        target_id: UUID | None = self._try_parse_uuid(uuid)
        if target_id is None:
            await session.out.send_system_message("parameter is not a valid UUID.")
            return
        try:
            self._db.set_branch_name(target_id, new_name)
        except db_err.InvalidInstanceError:
            await session.out.send_system_message(f"Failed to rename {uuid}")
            return
        await session.out.send_system_message(f"Branch {uuid} renamed {new_name}")

    async def _fork(self, session: Session) -> None:
        """Derive a new branch from the current one."""
        if session.conv.uuid is None or session.conv.model is None:
            await session.out.send_system_message("You cannot fork without a conversation started/setted")
            return
        uuid: UUID | None = self._db.create_branch(model=session.conv.model, current_branch=session.conv.uuid)
        if uuid is None:
            await session.out.send_system_message("Failed to make a new branch")
            return
        await self._set(session, str(uuid))

    @staticmethod
    async def _clear_cache(session: Session) -> None:
        """Clear the session's semantic cache, reporting a failure without raising it."""
        if session.cache is None:
            return
        try:
            await session.cache.clear()
        except CacheError as error:
            await session.out.send_system_message(f"The semantic cache could not be cleared: {error}")

    @override
    async def execute(self, session: Session, *args: str) -> None:
        """Dispatch to the subcommand named by the first argument, passing it the rest."""
        if not args:
            await session.out.send_system_message(f"Missing subcommand. Usage: {self.usage}")
            return
        name, *rest = args
        handler = self._subcommands.get(name)
        if handler is None:
            await session.out.send_system_message(f"Unknown subcommand '{name}'. Usage: {self.usage}")
            return
        try:
            await handler(session, *rest)
        except TypeError:
            await session.out.send_system_message(f"Invalid arguments for '{name}'. Usage: {self.usage}")

    @override
    def completions(self, session: Session, args: list[str]) -> list[str]:
        """Return the available subcommand names."""
        del session
        size: int = len(args)
        if size == 0:
            return sorted(self._subcommands)
        if size == 1 and args[0] in ["set", "remove", "rename"]:
            return list(map(str, self._db.get_branches().keys()))
        return []

    @staticmethod
    def _try_parse_uuid(value: str) -> UUID | None:
        try:
            return UUID(value)
        except ValueError, TypeError, AttributeError:
            return None
