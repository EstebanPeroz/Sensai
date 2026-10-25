from typing import Annotated, Literal

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


type ClientMessage = Annotated[Login | Logout | Chat | Command | Cancel | Complete, Field(discriminator="type")]
