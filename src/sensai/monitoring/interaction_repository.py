from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from datetime import datetime

    from sensai.monitoring.interaction import Interaction


class InteractionRepository(Protocol):
    """Storage of the logged calls to the model, independent of its backend."""

    def add(self, interaction: Interaction) -> None:
        """Store `interaction`, the record of one call to the model.

        Raises:
            DatabaseError: If it could not be stored.

        """

    def get_all(self, *, since: datetime | None = None, with_content: bool = True) -> list[Interaction]:
        """Return the stored interactions created at or after `since` (every one if None), oldest first.

        Without content, their request, response, thinking and tool calls are left empty: enough for
        metrics, and much lighter to read.

        Raises:
            DatabaseError: If they could not be read.

        """
        ...
