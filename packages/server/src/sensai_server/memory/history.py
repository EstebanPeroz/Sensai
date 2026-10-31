from __future__ import annotations

from typing import TYPE_CHECKING

from sensai_server.memory.message import Message

if TYPE_CHECKING:
    from sensai_protocol.server_messages import Role


class History:
    """Conversation history kept in memory, lost when the session ends."""

    def __init__(self) -> None:
        """Init an empty history."""
        self._messages: list[Message] = []

    def append(
        self,
        role: Role,
        content: str,
        *,
        tool_calls: list[dict] | None = None,
        tool_name: str | None = None,
    ) -> None:
        """Add a message at the end of the history, with the tool calls it makes or the tool it answers for."""
        self._messages.append(
            Message(uuid=None, role=role, content=content, tool_calls=tool_calls, tool_name=tool_name)
        )

    def insert(self, message: Message) -> None:
        """Add a message at the end of the history."""
        self._messages.append(message)

    def messages(self) -> list[Message]:
        """Return a copy of the messages, oldest first, so callers cannot alter the history."""
        return list(self._messages)

    def to_json(self) -> list[dict[str, object]]:
        """Return the messages in the format of the chat payload `messages` list, oldest first."""
        return [message.to_dict() for message in self._messages]

    def clear(self) -> None:
        """Remove every message from the history."""
        self._messages.clear()

    def replace(self, messages: list[Message]) -> None:
        """Remove every message from the history."""
        self.clear()
        self._messages = messages
