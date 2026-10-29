from typing import TYPE_CHECKING, override
from uuid import UUID

import sensai.memory.persistent_db.error as db_err
from sensai.commands.base import StateCommand

if TYPE_CHECKING:
    from collections.abc import Callable

    from sensai.history.conversation import Conversation
    from sensai.llm.provider_manager import ProviderManager


class Branch(StateCommand):
    """Branch command class to display or choose available models."""

    name = "branch"
    help = "Update the branch or load them"
    usage = "/branch list, branch set <uuid>, branch remove <uuid>, branch rename <uuid> <new_name>, branch fork"

    _subcommands: dict[str, Callable[..., None]]
    _conv: Conversation
    _pm: ProviderManager
    _refresh_context_usage: Callable[[], object]

    def __init__(
        self, conv: Conversation, provider_manager: ProviderManager, refresh_context_usage: Callable[[], object]
    ) -> None:
        """Initialize the Branch command and its subcommand map.

        `refresh_context_usage` is called once a branch is loaded, since the usage is kept per branch.

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
        self._conv = conv
        self._pm = provider_manager
        self._refresh_context_usage = refresh_context_usage

    def _list(self) -> None:
        """Display the available branches."""
        branches: dict[UUID, str] = self._fetch_branches()
        for i, (branch_id, name) in enumerate(branches.items()):
            self._ui.send_system_message(
                str(branch_id) + ": " + name,
                append_response=(i != 0),
            )

    def _set(self, uuid: str) -> None:
        """Load the branch identified by uuid."""
        target_id: UUID | None = self._try_parse_uuid(uuid)
        if target_id is None:
            self._ui.send_system_message("parameter is not a valid UUID.")
            return
        try:
            self._db.get_branch(
                current=self._conv,
                target_branch=target_id,
                provider_manager=self._pm,
            )
        except db_err.InvalidInstanceError as err:
            self._ui.send_system_message(str(err))
            return
        self._ui.load_conversation(self._conv.history)
        self._refresh_context_usage()

    def _remove(self, uuid: str) -> None:
        """Remove the branch identified by uuid."""
        target_id: UUID | None = self._try_parse_uuid(uuid)
        if target_id is None:
            self._ui.send_system_message("parameter is not a valid UUID.")
            return
        if target_id == self._conv.uuid:
            self._ui.send_system_message("You cannot remove your current branch.")
            return
        try:
            self._db.remove_branch(target_id)
        except db_err.InvalidInstanceError:
            self._ui.send_system_message(f"Failed to remove {uuid}")
            return
        self._ui.send_system_message(f"Branch {uuid} removed")

    def _rename(self, uuid: str, new_name: str) -> None:
        """Rename the branch identified by uuid."""
        target_id: UUID | None = self._try_parse_uuid(uuid)
        if target_id is None:
            self._ui.send_system_message("parameter is not a valid UUID.")
            return
        try:
            self._db.set_branch_name(target_id, new_name)
        except db_err.InvalidInstanceError:
            self._ui.send_system_message(f"Failed to rename {uuid}")
            return
        self._ui.send_system_message(f"Branch {uuid} renamed {new_name}")

    def _fork(self) -> None:
        """Derive a new branch from the current one."""
        if self._conv.uuid is None:
            self._ui.send_system_message("You cannot fork without a conversation started/setted")
            return
        uuid: UUID | None = self._db.create_branch(self._conv.uuid)
        if uuid is None:
            self._ui.send_system_message("Failed to make a new branch")
            return
        self._set(str(uuid))

    @override
    def execute(self, *args: str) -> None:
        """Dispatch to the subcommand named by the first argument, passing it the rest."""
        if not args:
            self._ui.send_system_message(f"Missing subcommand. Usage: {self.usage}")
            return
        name, *rest = args
        handler = self._subcommands.get(name)
        if handler is None:
            self._ui.send_system_message(f"Unknown subcommand '{name}'. Usage: {self.usage}")
            return
        try:
            handler(*rest)
        except TypeError:
            self._ui.send_system_message(f"Invalid arguments for '{name}'. Usage: {self.usage}")

    @override
    def completions(self, args: list[str]) -> list[str]:
        """Return the available subcommand names."""
        size: int = len(args)
        if size == 0:
            return sorted(self._subcommands)
        if size == 1 and args[0] in ["set", "remove", "rename"]:
            return self._uuid_completion()
        return []

    def _uuid_completion(self) -> list[str]:
        if self._cache_id == []:
            self._fetch_branches()
        return [str(uuid) for uuid in self._cache_id]

    @staticmethod
    def _try_parse_uuid(value: str) -> UUID | None:
        try:
            return UUID(value)
        except ValueError, TypeError, AttributeError:
            return None
