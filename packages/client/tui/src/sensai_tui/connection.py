from __future__ import annotations

from typing import TYPE_CHECKING

from sensai_protocol import Error, ProtocolError, parse_server_message
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sensai_protocol import ClientMessage, ServerMessage


class ConnectionFailedError(Exception):
    """The websocket to the server could not be opened."""


class Connection:
    """Websocket to the server: sends client messages and yields the parsed server messages."""

    url: str
    _ws: ClientConnection | None

    def __init__(self, url: str) -> None:
        """Remember the server `url`. Nothing is opened before `open`."""
        self._url = url
        self._ws = None

    async def open(self) -> None:
        """Connect to the server. Raise ConnectionFailedError if it cannot be reached."""
        try:
            self._ws = await connect(self._url)
        except (OSError, TimeoutError, InvalidURI, InvalidHandshake) as err:
            raise ConnectionFailedError(str(err) or type(err).__name__) from err

    async def send(self, message: ClientMessage) -> None:
        """Send `message`. On a closed socket it is dropped: `messages` reports the disconnection."""
        if self._ws is None:
            return
        try:
            await self._ws.send(message.model_dump_json())
        except ConnectionClosed:
            return

    async def messages(self) -> AsyncIterator[ServerMessage]:
        """Yield each message received, until the socket closes.

        A frame that is not a valid server message is turned into an `Error`, so it is shown instead of crashing.
        """
        if self._ws is None:
            return
        try:
            async for frame in self._ws:
                try:
                    yield parse_server_message(frame)
                except ProtocolError as err:
                    yield Error(message=f"protocol error: {err}")
        except ConnectionClosed:
            return

    async def close(self) -> None:
        """Close the socket, if open."""
        if self._ws is None:
            return
        ws, self._ws = self._ws, None
        await ws.close()
