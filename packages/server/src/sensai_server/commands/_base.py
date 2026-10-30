from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from sensai_server.session import Session


class Command(ABC):
    """A command the server runs for a client, such as `/persona list`.

    One instance is shared by every session: the session the command runs for is given to each call.
    """

    name: ClassVar[str]
    help: ClassVar[str]
    usage: ClassVar[str]

    @abstractmethod
    async def execute(self, session: Session, *args: str) -> None:
        """Execute the command with the given arguments, for `session`."""

    def completions(self, session: Session, args: list[str]) -> list[str]:
        """Return the values the next argument can take after the finished `args`, for completion."""
        del session, args
        return []
