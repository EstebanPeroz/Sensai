from pydantic import TypeAdapter, ValidationError

from sensai_protocol.client_messages import ClientMessage
from sensai_protocol.error import ProtocolError
from sensai_protocol.server_messages import ServerMessage

_client_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)
_server_adapter: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)


def parse_client_message(raw: str | bytes) -> ClientMessage:
    """Parse a frame sent by a client. Raise ProtocolError if it is not a valid client message."""
    try:
        return _client_adapter.validate_json(raw)
    except ValidationError as err:
        raise ProtocolError(str(err)) from err


def parse_server_message(raw: str | bytes) -> ServerMessage:
    """Parse a frame sent by the server. Raise ProtocolError if it is not a valid server message."""
    try:
        return _server_adapter.validate_json(raw)
    except ValidationError as err:
        raise ProtocolError(str(err)) from err
