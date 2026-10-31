from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from uuid import UUID


@dataclass(frozen=True)
class Summary:
    """Condensed form of the oldest messages of a branch, sent to the model in their place."""

    covered: int
    """Number of messages it stands for, from the first one of the branch."""
    content: str


class SummaryRepository(Protocol):
    """Storage of the summary of each branch, independent of its backend."""

    def get(self, branch_id: UUID) -> Summary | None:
        """Return the summary of the branch `branch_id`, None when it has none.

        Raises:
            DatabaseError: If it could not be read.

        """
        ...

    def save(self, branch_id: UUID, summary: Summary) -> None:
        """Make `summary` the summary of the branch `branch_id`, replacing the previous one.

        Raises:
            DatabaseError: If it could not be stored.

        """
