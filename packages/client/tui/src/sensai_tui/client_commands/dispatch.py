from __future__ import annotations

from typing import TYPE_CHECKING

from sensai_protocol import CommandContext, parse_params
from sensai_protocol.error import SensaiError

from sensai_tui.client_commands import CLIENT_COMMAND_HANDLERS, ClientCommandFailedError

if TYPE_CHECKING:
    from sensai_protocol import RequestCommand


async def run_client_command(message: RequestCommand) -> CommandContext:
    """Run the client command the server asked for, and return the answer to send back.

    It never raises: a missing handler, invalid params or a failing command are answered with an `error`.
    """
    try:
        handler = CLIENT_COMMAND_HANDLERS.get(message.command)
        if handler is None:
            return _failed(message, f"the client does not implement '{message.command}'")
        result = await handler(parse_params(message.command, message.params))
    except ClientCommandFailedError as err:
        return _failed(message, err.reason)
    except SensaiError as err:
        return _failed(message, err.message)
    except Exception as err:  # noqa: BLE001 - every failure is reported to the server, never raised
        return _failed(message, str(err) or type(err).__name__)
    return CommandContext(request_id=message.request_id, command=message.command, ctx=result.model_dump())


def _failed(message: RequestCommand, reason: str) -> CommandContext:
    """Build the answer to `message` reporting that it failed for `reason`."""
    return CommandContext(request_id=message.request_id, command=message.command, error=reason)
