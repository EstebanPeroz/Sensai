from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID  # noqa: TC003 - pydantic resolves annotations at runtime

from pydantic import Field

from sensai_protocol._base import ProtocolModel


class Login(ProtocolModel):
    """Open a session with `model`. Must be the first message of a connection."""

    type: Literal["login"] = "login"
    model: str


class Logout(ProtocolModel):
    """Close the session. The server then closes the socket."""

    type: Literal["logout"] = "logout"


class Chat(ProtocolModel):
    """Send a prompt to the conversation."""

    type: Literal["chat"] = "chat"
    prompt: str


class Command(ProtocolModel):
    """Run a command, such as `/model llama3:8b`."""

    type: Literal["command"] = "command"
    command: str


class Cancel(ProtocolModel):
    """Stop the task running on the session, if any."""

    type: Literal["cancel"] = "cancel"


class Complete(ProtocolModel):
    """Ask for the completion values of the current input."""

    type: Literal["complete"] = "complete"
    text: str


class Confirmation(StrEnum):
    """Answer of the user asked whether a tool may run: once, for the rest of the session, or not at all."""

    ALLOW = "allow"
    ALWAYS = "always"
    DENY = "deny"


class ConfirmAnswer(ProtocolModel):
    """Answer to the `confirm_tool` of `request_id`."""

    type: Literal["confirm_answer"] = "confirm_answer"
    request_id: UUID
    answer: Confirmation


class UserAnswer(ProtocolModel):
    """Answer to the `ask_user` of `request_id`. None when the user skipped the question."""

    type: Literal["user_answer"] = "user_answer"
    request_id: UUID
    answer: str | None


type ClientMessage = Annotated[
    Login | Logout | Chat | Command | Cancel | Complete | ConfirmAnswer | UserAnswer,
    Field(discriminator="type"),
]
