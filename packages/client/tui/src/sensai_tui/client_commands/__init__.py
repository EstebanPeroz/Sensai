from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sensai_protocol import ClientCommand

from .file import ClientCommandFailedError, read_file

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sensai_protocol import ProtocolModel

type ClientCommandHandler = Callable[[Any], Awaitable[ProtocolModel]]

CLIENT_COMMAND_HANDLERS: dict[ClientCommand, ClientCommandHandler] = {
    ClientCommand.READ_FILE: read_file,
}

__all__ = ["CLIENT_COMMAND_HANDLERS", "ClientCommandFailedError", "ClientCommandHandler", "read_file"]
