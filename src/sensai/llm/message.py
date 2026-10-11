from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    """Author of a chat message, as expected by the chat endpoint."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    """One role-tagged turn of a conversation, mapping 1:1 onto a chat message."""

    role: Role
    content: str
    tool_calls: list[dict] | None = None
    tool_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the message in the format of the chat payload `messages` list."""
        message: dict[str, object] = {"role": self.role.value, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        if self.tool_name is not None:
            message["tool_name"] = self.tool_name
        return message
