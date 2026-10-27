from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sensai.monitoring.interaction import Interaction


class InteractionRepository(Protocol):
    """Storage of the logged calls to the model, independent of its backend."""

    def add(self, interaction: Interaction) -> None:
        """Store `interaction`, the record of one call to the model.

        Raises:
            DatabaseError: If it could not be stored.

        """

    def get_all(self) -> list[Interaction]:
        """Return every stored interaction, oldest first."""
        ...
