from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from sensai.history.conversation import Conversation
    from sensai.llm.message import Message
    from sensai.llm.persona import Persona


class PersistentDatabase(ABC):
    """Persistent storage of conversation branches, their messages and personas, independent of its backend."""

    # Branch

    @abstractmethod
    def create_branch(self, current_branch: UUID | None = None) -> UUID | None:
        """Create a new branch and return its id if success or None if fail."""

    @abstractmethod
    def get_branch_list(self) -> dict[str, UUID]:
        """Return the dict of all branches with name and uuid."""

    @abstractmethod
    def get_branch(self, current_branch: Conversation, target_branch: UUID) -> None:
        """Replace the conversation info with those of the target.

        Raises:
            KeyError: If no branch has this uuid.

        """

    @abstractmethod
    def remove_branch(self, uuid: UUID) -> None:
        """Remove the branch related to this uuid.

        Raises:
            KeyError: If no branch has this uuid.

        """

    @abstractmethod
    def add_message_to_branch(self, branch_id: UUID, message: Message) -> None:
        """Append `message` at the end of the branch `branch_id`."""

    @abstractmethod
    def change_branch_persona(self, branch: UUID, persona: UUID) -> bool:
        """Set the persona used by the branch `branch` to `persona`.

        True = Success
        False = Fail

        """

    # Message

    @abstractmethod
    def get_messages(self, branch: UUID) -> list[Message]:
        """Return every message of the branch `branch`, oldest first."""

    # Persona

    @abstractmethod
    def create_persona(self, name: str, description: str, timestamp: datetime, prompt: str) -> Persona | None:
        """Create a persona and return its Class Version after adding it to the db.

        Args:
            name: Display name of the persona.
            description: Short summary of what the persona is for.
            timestamp: Creation time of the persona.
            prompt: System prompt applied when the persona is active.

        """

    @abstractmethod
    def get_personas(self) -> list[Persona]:
        """Return all personas."""

    @abstractmethod
    def get_persona(self, persona: UUID) -> Persona:
        """Return the persona `persona`.

        Raises:
            KeyError: If no persona has this id.

        """

    @abstractmethod
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
