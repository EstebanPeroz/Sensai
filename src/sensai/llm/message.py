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

    def to_dict(self) -> dict[str, str]:
        """Return the message in the format of the chat payload `messages` list."""
        return {"role": self.role.value, "content": self.content}
