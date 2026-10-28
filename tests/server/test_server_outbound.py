import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import pytest
from fastapi import WebSocket, WebSocketDisconnect
from sensai_protocol import (
    AiChunk,
    AskUser,
    Completions,
    Confirmation,
    ConfirmTool,
    Dashboard,
    DashboardReport,
    DashboardSummary,
    Effort,
    Error,
    History,
    HistoryMessage,
    Input,
    Period,
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
SUMMARY = DashboardSummary(
    calls=0,
    errors=0,
    interrupted=0,
    latency_p50_ms=None,
    latency_p95_ms=None,
    first_token_p50_ms=None,
    tokens_per_s=None,
    prompt_tokens=0,
    completion_tokens=0,
)
REPORT = DashboardReport(
    period=Period.DAY,
    start=datetime(2026, 10, 13, tzinfo=UTC),
    end=datetime(2026, 10, 14, tzinfo=UTC),
    summary=SUMMARY,
    calls_series=[],
    latency_series=[],
    groups=[],
    drifts=[],
    recent_errors=[],
    baseline_periods=7,
)

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
    (lambda out: out.show_dashboard(REPORT), Dashboard(report=REPORT)),
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


async def _asked(ws: FakeWebSocket) -> ServerMessage:
    """Let the asking task run until it waits for its answer, and return the question it sent."""
    await asyncio.sleep(0)
    return parse_server_message(ws.frames[-1])


class TestQuestions:
    async def test_confirmation_waits_for_its_answer(self) -> None:
        ws = FakeWebSocket()
        out = _outbound(ws)
        task = asyncio.create_task(out.ask_confirmation("web_search", {"query": "paris"}))
        request = await _asked(ws)
        assert isinstance(request, ConfirmTool)
        assert (request.tool_name, request.arguments) == ("web_search", {"query": "paris"})

        out.resolve(uuid4(), Confirmation.ALLOW)
        await asyncio.sleep(0)
        assert not task.done()
        out.resolve(request.request_id, Confirmation.ALWAYS)
        assert await task == Confirmation.ALWAYS
        assert not out.waiting

    async def test_ask_user_shows_question_and_answer(self) -> None:
        ws = FakeWebSocket()
        out = _outbound(ws)
        task = asyncio.create_task(out.ask_user("Which city?"))
        request = await _asked(ws)
        assert isinstance(request, AskUser)
        out.resolve(request.request_id, None)
        assert await task is None
        assert parse_server_message(ws.frames[-1]) == System(message="Sensai asked: Which city?\n> (no answer)")

    async def test_closed_socket_denies_and_skips(self) -> None:
        out = _outbound(FakeWebSocket(RuntimeError("closed")))
        assert await out.ask_confirmation("now", {}) == Confirmation.DENY
        assert await out.ask_user("Which city?") is None
        assert not out.waiting

    async def test_cancel_forgets_the_question(self) -> None:
        ws = FakeWebSocket()
        out = _outbound(ws)
        task = asyncio.create_task(out.ask_confirmation("now", {}))
        request = await _asked(ws)
        task.cancel()
        await asyncio.wait({task})
        assert not out.waiting
        assert isinstance(request, ConfirmTool)
        out.resolve(request.request_id, Confirmation.ALLOW)
