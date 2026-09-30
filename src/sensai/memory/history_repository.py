from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sensai.llm.message import Message, Role


class HistoryRepository(Protocol):
    """Storage of the conversation history of a session, independent of its backend."""

    def append(self, role: Role, content: str) -> None:
        """Add a message at the end of the history."""

    def messages(self) -> list[Message]:
        """Return the messages of the history, oldest first."""
        ...

    def to_json(self) -> list[dict[str, str]]:
        """Return the messages in the format of the chat payload `messages` list, oldest first."""
        ...

    def clear(self) -> None:
        """Remove every message from the history."""
