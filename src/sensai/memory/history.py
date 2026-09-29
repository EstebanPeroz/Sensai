from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sensai.llm.message import Message


class InMemoryHistory:
    """Conversation history kept in memory, lost when the session ends."""

    def __init__(self) -> None:
        """Init an empty history."""
        self._messages: list[Message] = []

    def append(self, message: Message) -> None:
        """Add a message at the end of the history."""
        self._messages.append(message)

    def messages(self) -> list[Message]:
        """Return a copy of the messages, oldest first, so callers cannot alter the history."""
        return list(self._messages)

    def clear(self) -> None:
        """Remove every message from the history."""
        self._messages.clear()
