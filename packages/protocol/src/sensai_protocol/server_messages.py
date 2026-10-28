from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID  # noqa: TC003 - pydantic resolves annotations at runtime

from pydantic import Field

from sensai_protocol._base import ProtocolModel
from sensai_protocol.dashboard import DashboardReport  # noqa: TC001 - pydantic resolves annotations at runtime


class Role(StrEnum):
    """Author of a chat message, as expected by the chat endpoint."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class HistoryMessage(ProtocolModel):
    """One message of a loaded conversation."""

    uuid: UUID | None
    role: Role
    content: str


class Session(ProtocolModel):
    """Confirm a login, with everything the client needs to draw its UI."""

    type: Literal["session"] = "session"
    client_id: UUID
    model: str
    commands: list[str]
    effort: str


class AiChunk(ProtocolModel):
    """One streamed chunk of an answer. The last one has `done` set; `error` is set on a provider error."""

    type: Literal["ai_chunk"] = "ai_chunk"
    content: str | None = None
    thinking: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    done: bool = False
    error: str | None = None


class Input(ProtocolModel):
    """Echo of what the user sent, as the server received it."""

    type: Literal["input"] = "input"
    role: Role
    content: str
    command: bool = False


class System(ProtocolModel):
    """A command output or application message. With `append`, it extends the previous system message."""

    type: Literal["system"] = "system"
    message: str
    append: bool = False


class Error(ProtocolModel):
    """An error to display."""

    type: Literal["error"] = "error"
    message: str


class History(ProtocolModel):
    """A loaded conversation. The client replaces its transcript with these messages."""

    type: Literal["history"] = "history"
    messages: list[HistoryMessage]


class Effort(ProtocolModel):
    """The thinking effort changed."""

    type: Literal["effort"] = "effort"
    level: str


class Completions(ProtocolModel):
    """Answer to a `complete`. It repeats `text`, so the client can drop an answer to an outdated input."""

    type: Literal["completions"] = "completions"
    text: str
    values: list[str]


class ConfirmTool(ProtocolModel):
    """Ask whether the tool `tool_name` may run with `arguments`. Answered by a `confirm_answer`."""

    type: Literal["confirm_tool"] = "confirm_tool"
    request_id: UUID
    tool_name: str
    arguments: dict[str, Any]


class AskUser(ProtocolModel):
    """Ask the user `question` on behalf of the model. Answered by a `user_answer`."""

    type: Literal["ask_user"] = "ask_user"
    request_id: UUID
    question: str


class Dashboard(ProtocolModel):
    """Show a monitoring report until the user closes it."""

    type: Literal["dashboard"] = "dashboard"
    report: DashboardReport


type ServerMessage = Annotated[
    Session | AiChunk | Input | System | Error | History | Effort | Completions | ConfirmTool | AskUser | Dashboard,
    Field(discriminator="type"),
]
