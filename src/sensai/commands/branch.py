from typing import TYPE_CHECKING, override
from uuid import UUID

import sensai.memory.persistent_db.error as db_err
from sensai.commands.base import Command
from sensai.memory.persistent_db import get_db

if TYPE_CHECKING:
    from collections.abc import Callable

    from sensai.history.conversation import Conversation
    from sensai.ui.adapter import UIAdapter


class Branch(Command):
    """Branch command class to display or choose available models."""

    name = "branch"
    help = "Update the branch or load them"
    usage = "/branch list, branch set <uuid>, branch remove <uuid>, branch rename <uuid> <new_name>, branch derive"

    _subcommands: dict[str, Callable[..., None]]
    _conversation: Conversation

    def __init__(self, ui: UIAdapter, conv: Conversation) -> None:
        """Initialize the Branch command and its subcommand map."""
        super().__init__(ui)
        self._subcommands = {
            "list": self._list,
            "set": self._set,
            "remove": self._remove,
            "rename": self._rename,
            "derive": self._derive,
        }
        self._conversation = conv

    def _list(self) -> None:
        """Display the available branches."""
        db = get_db()
        if not db:
            self._ui.send_system_message("Fail to access db.")
            return
        branches: dict[UUID, str] = db.get_branches()
        for i, (branch_id, name) in enumerate(branches.items()):
            self._ui.send_system_message(
                str(branch_id) + ": " + name,
                append_response=(i != 0),
            )

    def _set(self, uuid: str) -> None:
        """Load the branch identified by uuid."""
        db = get_db()
        if not db:
            self._ui.send_system_message("Fail to access db.")
            return
        target_id: UUID | None = self._try_parse_uuid(uuid)
        if target_id is None:
            self._ui.send_system_message("parameter is not a valid UUID.")
            return
        try:
            db.get_branch(
                current=self._conversation,
                target_branch=target_id,
                provider_manager=self._conversation._provider_manager,
            )
        except db_err.InvalidInstanceError as err:
            self._ui.send_system_message(str(err))
            return
        self._ui.load_conversation(self._conversation.history)

    def _remove(self, uuid: str) -> None:
        """Remove the branch identified by uuid."""
        target_id: UUID | None = self._try_parse_uuid(uuid)
        if target_id is None:
            self._ui.send_system_message("parameter is not a valid UUID.")
            return
        if uuid == self._conversation.uuid:
            self._ui.send_system_message("You cannot remove your current branch.")
            return
        db = get_db()
        if not db:
            self._ui.send_system_message("Fail to access db.")
            return
        db.remove_branch()

    def _rename(self, uuid: str, new_name: str) -> None:
        """Rename the branch identified by uuid."""
        if uuid == self._conversation.uuid:
            self._ui.send_system_message("You cannot remove your current branch.")
            return
        db = get_db()
        if not db:
            self._ui.send_system_message("Fail to access db.")
            return

    def _derive(self) -> None:
        """Derive a new branch from the current one."""
        self._ui.send_system_message("derive")

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
    def completions(self) -> list[str]:
        """Return the available subcommand names."""
        return sorted(self._subcommands)

    @staticmethod
    def _try_parse_uuid(value: str) -> UUID | None:
        try:
            return UUID(value)
        except ValueError, TypeError, AttributeError:
            return None
