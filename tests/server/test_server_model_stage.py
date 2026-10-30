import asyncio
import threading
import warnings
from collections.abc import Iterator
from contextlib import aclosing

import pytest
from sensai_protocol.error import SensaiError
from sensai_server.llm.error import RequestCallError
from sensai_server.llm.provider import ProviderManager
from sensai_server.pipeline import ChatContext
from sensai_server.pipeline.modules import ModelStage
from sensai_server.pipeline.modules._threaded import iterate_in_thread

from tests.server.conftest import MODEL, SETTINGS
from tests.server.fakes import FakeAdapter, RecordingOutbound, chunk

MESSAGES = [{"role": "user", "content": "hi"}]
TIMEOUT = 2


def _context(model: str = MODEL) -> ChatContext:
    return ChatContext(message="hi", out=RecordingOutbound(), model=model, messages=list(MESSAGES))


async def _contents(context: ChatContext) -> list[str | None]:
    async with aclosing(ModelStage(ProviderManager(SETTINGS.llm))(context)) as chunks:
        return [item.content async for item in chunks]


async def _wait(event: threading.Event) -> bool:
    """Wait for `event`, set from another thread, without blocking the loop."""
    return await asyncio.to_thread(event.wait, TIMEOUT)


class TestModelStage:
    async def test_streams_chunks_in_order(self) -> None:
        FakeAdapter.replies = [chunk("a"), chunk("b"), chunk("c", done=True)]
        assert await _contents(_context()) == ["a", "b", "c"]
        assert await _wait(FakeAdapter.closed)

    async def test_payload(self) -> None:
        FakeAdapter.replies = [chunk("a", done=True)]
        context = _context()
        await _contents(context)
        payload = FakeAdapter.last_payload
        assert payload is not None
        assert payload == {"model": MODEL, "messages": MESSAGES}
        assert payload["messages"] is not context.messages

    async def test_unknown_model(self) -> None:
        with pytest.raises(SensaiError, match="No provider serves model nope"):
            await _contents(_context("nope"))

    async def test_loop_stays_responsive(self) -> None:
        FakeAdapter.block = threading.Event()
        FakeAdapter.replies = [chunk("a"), chunk("b", done=True)]
        stream = asyncio.create_task(_contents(_context()))
        ticks = 0
        for _ in range(10):
            await asyncio.sleep(0)
            ticks += 1
        assert ticks == 10
        assert not stream.done()
        FakeAdapter.block.set()
        assert await asyncio.wait_for(stream, TIMEOUT) == ["a", "b"]

    async def test_cancel_closes_provider_stream(self) -> None:
        FakeAdapter.block = threading.Event()
        FakeAdapter.replies = [chunk("a"), chunk("b", done=True)]
        first = asyncio.Event()

        async def consume() -> None:
            async with aclosing(ModelStage(ProviderManager(SETTINGS.llm))(_context())) as chunks:
                async for _ in chunks:
                    first.set()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            task = asyncio.create_task(consume())
            await asyncio.wait_for(first.wait(), TIMEOUT)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not FakeAdapter.closed.is_set()
            # The thread is still inside `next()`: the close runs once it returns, without a `ValueError`.
            FakeAdapter.block.set()
            assert await _wait(FakeAdapter.closed)

    async def test_provider_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def failing_chat(_self: FakeAdapter, _payload: dict, *, stream: bool) -> Iterator:  # noqa: ARG001
            msg = "connection refused"
            raise RequestCallError(msg)

        monkeypatch.setattr(FakeAdapter, "chat", failing_chat)
        with pytest.raises(RequestCallError):
            await _contents(_context())


async def _drain(iterator: Iterator[int]) -> list[int]:
    async with aclosing(iterate_in_thread(lambda: iterator)) as items:
        return [item async for item in items]


class TestIterateInThread:
    async def test_early_close_closes_iterator(self) -> None:
        closed = threading.Event()

        def numbers() -> Iterator[int]:
            try:
                yield from range(10)
            finally:
                closed.set()

        async with aclosing(iterate_in_thread(numbers)) as items:
            async for item in items:
                assert item == 0
                break
        assert await _wait(closed)

    async def test_error_in_iterator_propagates(self) -> None:
        def broken() -> Iterator[int]:
            yield 1
            msg = "boom"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="boom"):
            await _drain(broken())

    async def test_plain_iterator_without_close(self) -> None:
        assert await _drain(iter([1, 2])) == [1, 2]
