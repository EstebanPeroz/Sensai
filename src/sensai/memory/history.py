from __future__ import annotations

from typing import TYPE_CHECKING

from sensai.llm.message import Message

if TYPE_CHECKING:
    from sensai.llm.message import Role


class History:
    """Conversation history kept in memory, lost when the session ends."""

    def __init__(self) -> None:
        """Init an empty history."""
        self._messages: list[Message] = []

    def append(self, role: Role, content: str) -> None:
        """Add a message at the end of the history."""
        self._messages.append(Message(role=role, content=content))

    def messages(self) -> list[Message]:
        """Return a copy of the messages, oldest first, so callers cannot alter the history."""
        return self._messages

    def to_json(self) -> list[dict[str, str]]:
        """Return the messages in the format of the chat payload `messages` list, oldest first."""
        return [message.to_dict() for message in self._messages]

    def clear(self) -> None:
        """Remove every message from the history."""
        self._messages.clear()

    def replace(self, messages: list[Message]) -> None:
        """Remove every message from the history."""
        self.clear()
        self._messages = messages
