from typing import TYPE_CHECKING

from sqlalchemy import Connection, Engine, create_engine
from sqlalchemy.exc import ArgumentError

from sensai.llm.persona import Persona
from sensai.memory.persistent_db.adapter import PersistentDatabase
from sensai.memory.persistent_db.error import DbConnectionError
from sensai.memory.persistent_db.sql_alchemy.table import Base

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from sensai.history.conversation import Conversation
    from sensai.llm.message import Message


class SqlAlchemy(PersistentDatabase):
    """Tmp."""

    _engine: Engine
    _conn: Connection

    def __init__(self) -> None:
        """Tmp."""
        super().__init__()

        try:
            self._engine = create_engine("sqlite:///test.db")
        except ArgumentError:
            msg = "Invalid url to database"
            raise DbConnectionError(msg) from None
        self._conn = self._engine.connect()
        Base.metadata.create_all(self._engine)

    def create_branch(self, current_branch: UUID | None = None) -> UUID | None:
        """Create a new branch and return its id if success or None if fail."""
        return None

    def get_branch_list(self) -> dict[str, UUID]:
        """Return the dict of all branches with name and uuid."""
        return {}

    def get_branch(self, current_branch: Conversation, target_branch: UUID) -> None:
        """Replace the conversation info with those of the target.

        Raises:
            KeyError: If no branch has this uuid.

        """

    def remove_branch(self, uuid: UUID) -> None:
        """Remove the branch related to this uuid.

        Raises:
            KeyError: If no branch has this uuid.

        """

    def add_message_to_branch(self, branch: UUID, message: Message) -> None:
        """Append `message` at the end of the branch `branch`."""

    def change_branch_persona(self, branch: UUID, persona: UUID) -> bool:
        """Set the persona used by the branch `branch` to `persona`.

        True = Success
        False = Fail

        """
        return False

    def get_messages(self, branch: UUID) -> list[Message]:
        """Return every message of the branch `branch`, oldest first."""
        return []

    def create_persona(self, name: str, description: str, timestamp: datetime, prompt: str) -> Persona | None:
        """Create a persona and return its Class Version after adding it to the db.

        Args:
            name: Display name of the persona.
            description: Short summary of what the persona is for.
            timestamp: Creation time of the persona.
            prompt: System prompt applied when the persona is active.

        """
        return None

    def get_personas(self) -> list[Persona]:
        """Return all personas."""
        return []

    def get_persona(self, persona: UUID) -> Persona:
        """Return the persona `persona`.

        Raises:
            KeyError: If no persona has this id.

        """
        return Persona("", "")

    def update_persona(
        self,
        persona: UUID,
        *,
        name: str | None = None,
        description: str | None = None,
        prompt: str | None = None,
    ) -> None:
        """Update the persona `persona`; fields left to `None` are unchanged.

        Raises:
            KeyError: If no persona has this id.

        """
