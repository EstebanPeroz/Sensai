from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from uuid import UUID

    from sensai.memory.persistent_db.adapter import PersistentDatabase
    from sensai.ui.adapter import UIAdapter


class Command(ABC):
    """Command class abstract base class for all commands."""

    name: ClassVar[str]
    help: ClassVar[str]
    _ui: ClassVar[UIAdapter]

    @classmethod
    def bind_ui(cls, ui: UIAdapter) -> None:
        """Set the UI shared by every command."""
        Command._ui = ui

    @abstractmethod
    def execute(self, *args: str) -> None:
        """Execute the command with the given arguments."""

    def completions(self, args: list[str]) -> list[str]:
        """Return the values the command's argument can take, for completion."""
        del args
        return []


class StateCommand(Command):
    """Command updating the state of the conversation or interacting with the db."""

    _db: ClassVar[PersistentDatabase]
    _cache_id: ClassVar[list[UUID]] = []

    @classmethod
    def bind_db(cls, db: PersistentDatabase) -> None:
        """Set the db shared by every command."""
        StateCommand._db = db

    def _fetch_branches(self) -> dict[UUID, str]:
        self._cache_id.clear()
        branches: dict[UUID, str] = self._db.get_branches()
        for branch_id in branches:
            self._cache_id.append(branch_id)
        return branches
