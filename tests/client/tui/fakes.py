import asyncio
from collections.abc import AsyncIterator

from sensai_protocol import ClientMessage, ServerMessage
from sensai_tui.connection import Connection, ConnectionFailedError


class FakeConnection(Connection):
    """Records the messages the app sends, and lets a test inject the server's."""

    def __init__(self, *, fail_open: bool = False) -> None:
        super().__init__("ws://fake/ws")
        self.sent: list[ClientMessage] = []
        self.opened = False
        self.closed = False
        self._fail_open = fail_open
        self._inbox: asyncio.Queue[ServerMessage | None] = asyncio.Queue()

    async def open(self) -> None:
        if self._fail_open:
            msg = "connection refused"
            raise ConnectionFailedError(msg)
        self.opened = True

    async def send(self, message: ClientMessage) -> None:
        self.sent.append(message)

    async def messages(self) -> AsyncIterator[ServerMessage]:
        while (message := await self._inbox.get()) is not None:
            yield message

    async def close(self) -> None:
        self.closed = True

    def push(self, *messages: ServerMessage) -> None:
        for message in messages:
            self._inbox.put_nowait(message)

    def disconnect(self) -> None:
        self._inbox.put_nowait(None)
