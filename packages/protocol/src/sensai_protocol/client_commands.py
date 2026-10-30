from enum import StrEnum
from typing import Any

from pydantic import Field, ValidationError

from sensai_protocol._base import ProtocolModel
from sensai_protocol.error import ProtocolError

MAX_FILE_SIZE = 1_000_000  # characters a client command may send back as a file's content


class ClientCommand(StrEnum):
    """The commands the server can ask the client to run. The client must implement every one."""

    READ_FILE = "read_file"


class ReadFileParams(ProtocolModel):
    """Parameters of `read_file`: the path of the file, on the client's machine."""

    path: str = Field(min_length=1)


class FileContent(ProtocolModel):
    """Context of `read_file`: the whole content of the file, as text. Not parsed."""

    content: str = Field(max_length=MAX_FILE_SIZE)


CLIENT_COMMANDS: dict[ClientCommand, tuple[type[ProtocolModel], type[ProtocolModel]]] = {
    ClientCommand.READ_FILE: (ReadFileParams, FileContent),
}


def parse_params(command: ClientCommand, params: dict[str, Any]) -> ProtocolModel:
    """Check the `params` of a `request_command` for `command`. Raise ProtocolError if they do not match its model."""
    model, _ = CLIENT_COMMANDS[command]
    return _validate(model, params)


def parse_ctx(command: ClientCommand, ctx: dict[str, Any] | None) -> ProtocolModel:
    """Check the `ctx` of a `command_context` for `command`. Raise ProtocolError if it does not match its model."""
    if ctx is None:
        msg = f"no context for the client command '{command}'"
        raise ProtocolError(msg)
    _, model = CLIENT_COMMANDS[command]
    return _validate(model, ctx)


def _validate(model: type[ProtocolModel], data: dict[str, Any]) -> ProtocolModel:
    """Build `model` from `data`, raising ProtocolError if it does not match."""
    try:
        return model.model_validate(data)
    except ValidationError as err:
        raise ProtocolError(str(err)) from err
