from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from sensai.history.conversation import Conversation
    from sensai.llm.message import Message, Role
    from sensai.llm.persona import Persona
    from sensai.llm.provider_manager import ProviderManager
    from sensai.memory.summary import SummaryRepository
    from sensai.monitoring.interaction_repository import InteractionRepository


class PersistentDatabase(ABC):
    """Persistent storage of conversation branches, their messages and personas, independent of its backend."""

    # Branch

    @abstractmethod
    def create_branch(self, current_branch: UUID | None = None, name: str = "Branch") -> UUID | None:
        """Create a new branch and return its id if success or None if fail.

        If `current_branch` is given, the new branch is a deep copy of it (messages included)
        and takes its name; otherwise an empty branch named `name` is made. Branch names are
        unique: a taken name gets a numeric suffix (`name 2`, `name 3`, ...). Returns None
        if `current_branch` does not exist.
        """

    @abstractmethod
    def import_branch(self, config: dict) -> None:
        """Make a branch based on a config."""

    @abstractmethod
    def get_branches(self) -> dict[UUID, str]:
        """Return the dict of all branches with name and uuid."""

    @abstractmethod
    def get_branch(self, current: Conversation, target_branch: UUID, provider_manager: ProviderManager) -> None:
        """Replace the uuid, model and history of `current` with those of the target branch.

        Does nothing if no branch has this uuid.
        """

    @abstractmethod
    def remove_branch(self, uuid: UUID) -> None:
        """Remove the branch related to this uuid.

        Raises:
            InvalidInstanceError: If no branch has this uuid.

        """

    @abstractmethod
    def add_message_to_branch(self, branch_id: UUID, role: Role, content: str) -> UUID | None:
        """Append a message made of `role` and `content` at the end of the branch `branch_id`.

        Return the id of the new message, or None if it could not be saved.

        Raises:
            InvalidInstanceError: If no branch has this uuid.

        """

    @abstractmethod
    def set_branch_model(self, branch: UUID, model: str) -> None:
        """Set the model used by the branch `branch` to `model`.

        Raises:
            InvalidInstanceError: If no branch has this uuid.

        """

    @abstractmethod
    def set_branch_name(self, branch: UUID, name: str) -> None:
        """Set the name used by the branch `branch`.

        Raises:
            InvalidInstanceError: If no branch has this uuid.

        """

    @abstractmethod
    def set_branch_persona(self, branch: UUID, persona: UUID) -> bool:
        """Set the persona used by the branch `branch` to `persona`.

        Return True on success, False if the branch or the persona does not exist
        or the change could not be saved.

        """

    @abstractmethod
    def export_branch(self, branch: UUID) -> dict | None:
        """Export the branch based on the uuid."""

    # Message

    @abstractmethod
    def get_messages(self, branch: UUID) -> list[Message]:
        """Return every message of the branch `branch`, oldest first."""

    # Persona

    @abstractmethod
    def create_persona(self, name: str, description: str, timestamp: datetime, prompt: str) -> Persona | None:
        """Create a persona and return its Class Version after adding it to the db.

        Return None if the persona could not be created.

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
            InvalidInstanceError: If no persona has this id.

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
            InvalidInstanceError: If no persona has this id.

        """

    @property
    @abstractmethod
    def interactions(self) -> InteractionRepository:
        """Logged calls to the model, stored alongside the rest of the persistent data."""

    @property
    @abstractmethod
    def summaries(self) -> SummaryRepository:
        """Summaries of the oldest messages of each branch, stored alongside the branches."""
