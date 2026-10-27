from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect
from sensai_protocol import Completions, Effort, Error, History, Input, System

if TYPE_CHECKING:
    from sensai_protocol import AiChunk, HistoryMessage, Role, ServerMessage


class Outbound:
    """Sends the server messages of one connection to its socket.

    Replaces `UIAdapter` on the server side and keeps its method names, so the code copied from `src/` keeps its
    shape.
    """

    _ws: WebSocket

    def __init__(self, ws: WebSocket) -> None:
        """Send on `ws`, an accepted websocket."""
        self._ws = ws

    async def send(self, message: ServerMessage) -> None:
        """Send `message`. On a closed socket it is dropped: the receive loop sees the disconnection."""
        try:
            await self._ws.send_text(message.model_dump_json())
        except WebSocketDisconnect, RuntimeError:
            return

    async def send_ai_response(self, chunk: AiChunk) -> None:
        """Send a chunk of the streamed answer."""
        await self.send(chunk)

    async def send_error(self, message: str) -> None:
        """Send an error to display."""
        await self.send(Error(message=message))

    async def send_input(self, role: Role, content: str, *, command: bool = False) -> None:
        """Echo an input of the conversation, or a typed `command`."""
        await self.send(Input(role=role, content=content, command=command))

    async def send_system_message(self, message: str, *, append_response: bool = False) -> None:
        """Send an application message, such as a command output, appended to the previous one if asked."""
        await self.send(System(message=message, append=append_response))

    async def load_conversation(self, messages: list[HistoryMessage]) -> None:
        """Send a conversation that replaces the client's transcript."""
        await self.send(History(messages=messages))

    async def update_thinking_effort(self, level: str) -> None:
        """Send the current thinking effort `level` of the model."""
        await self.send(Effort(level=level))

    async def send_completions(self, text: str, values: list[str]) -> None:
        """Answer a `complete` of `text` with its completion `values`."""
        await self.send(Completions(text=text, values=values))
