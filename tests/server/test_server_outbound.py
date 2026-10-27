from collections.abc import Awaitable, Callable
from typing import cast
from uuid import uuid4

import pytest
from fastapi import WebSocket, WebSocketDisconnect
from sensai_protocol import (
    AiChunk,
    Completions,
    Effort,
    Error,
    History,
    HistoryMessage,
    Input,
    Role,
    ServerMessage,
    System,
    parse_server_message,
)
from sensai_server.outbound import Outbound


class FakeWebSocket:
    """Records the frames sent, or raises `error` on each send."""

    def __init__(self, error: Exception | None = None) -> None:
        self.frames: list[str] = []
        self._error = error

    async def send_text(self, data: str) -> None:
        if self._error is not None:
            raise self._error
        self.frames.append(data)


def _outbound(ws: FakeWebSocket) -> Outbound:
    return Outbound(cast("WebSocket", ws))


HISTORY = [HistoryMessage(uuid=uuid4(), role=Role.USER, content="hi")]
CHUNK = AiChunk(content="hel", done=False)

CASES: list[tuple[Callable[[Outbound], Awaitable[None]], ServerMessage]] = [
    (lambda out: out.send_ai_response(CHUNK), CHUNK),
    (lambda out: out.send_error("boom"), Error(message="boom")),
    (lambda out: out.send_input(Role.USER, "hi"), Input(role=Role.USER, content="hi")),
    (
        lambda out: out.send_input(Role.USER, "/help", command=True),
        Input(role=Role.USER, content="/help", command=True),
    ),
    (lambda out: out.send_system_message("done"), System(message="done")),
    (lambda out: out.send_system_message("more", append_response=True), System(message="more", append=True)),
    (lambda out: out.load_conversation(HISTORY), History(messages=HISTORY)),
    (lambda out: out.update_thinking_effort("high"), Effort(level="high")),
    (lambda out: out.send_completions("/model ", ["a", "b"]), Completions(text="/model ", values=["a", "b"])),
]


class TestOutbound:
    @pytest.mark.parametrize(("call", "expected"), CASES)
    async def test_sends_message(self, call: Callable[[Outbound], Awaitable[None]], expected: ServerMessage) -> None:
        ws = FakeWebSocket()
        await call(_outbound(ws))
        assert [parse_server_message(frame) for frame in ws.frames] == [expected]

    @pytest.mark.parametrize("error", [RuntimeError("closed"), WebSocketDisconnect()])
    async def test_closed_socket_is_swallowed(self, error: Exception) -> None:
        await _outbound(FakeWebSocket(error)).send_error("boom")
