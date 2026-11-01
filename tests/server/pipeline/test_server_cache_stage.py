import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import cast

import pytest
from sensai_protocol import System
from sensai_server.cache import CacheError, SessionCache
from sensai_server.llm.error import EmbeddingError
from sensai_server.llm.responses import ChatResponse
from sensai_server.pipeline import ChatContext, answer_chunk
from sensai_server.pipeline.modules import CacheStage

from tests.server.fakes import FakeSessionCache, RecordingNext, RecordingOutbound

PROMPT = "what is the engine?"
CACHED = "The cached answer."


def _chunk(content: str, *, done: bool = False) -> ChatResponse:
    return ChatResponse({"done": done, "message": {"role": "assistant", "content": content}})


def _context(cache: FakeSessionCache | None) -> ChatContext:
    return ChatContext(message=PROMPT, out=RecordingOutbound(), model="m", cache=cast("SessionCache | None", cache))


async def _run(context: ChatContext, next_stage: RecordingNext) -> list[ChatResponse]:
    async with aclosing(CacheStage()(context, next_stage)) as chunks:
        return [chunk async for chunk in chunks]


def _contents(chunks: list[ChatResponse]) -> list[str | None]:
    return [chunk.content for chunk in chunks]


def _system_messages(context: ChatContext) -> list[str]:
    out = cast("RecordingOutbound", context.out)
    return [message.message for message in out.sent if isinstance(message, System)]


class TestNoCache:
    async def test_relays_the_chain(self) -> None:
        chunks = (_chunk("a"), _chunk("b", done=True))
        next_stage = RecordingNext(*chunks)
        assert await _run(_context(None), next_stage) == list(chunks)
        assert len(next_stage.calls) == 1


class TestHit:
    async def test_answers_without_calling_the_chain(self) -> None:
        cache = FakeSessionCache({PROMPT: CACHED})
        next_stage = RecordingNext(_chunk("model", done=True))

        chunks = await _run(_context(cache), next_stage)

        assert _contents(chunks) == [CACHED]
        assert chunks[0].done
        assert next_stage.calls == []

    async def test_stores_nothing(self) -> None:
        cache = FakeSessionCache({PROMPT: CACHED})
        await _run(_context(cache), RecordingNext())
        assert cache.stored == []


class TestMiss:
    async def test_streams_the_chain_and_stores_once_on_done(self) -> None:
        cache = FakeSessionCache()
        chunks = (_chunk("Hel"), _chunk("lo"), _chunk("", done=True))
        next_stage = RecordingNext(*chunks)

        assert await _run(_context(cache), next_stage) == list(chunks)

        assert cache.lookups == [PROMPT]
        assert cache.stored == [(PROMPT, "Hello")]

    async def test_an_errored_answer_is_not_stored(self) -> None:
        cache = FakeSessionCache()
        await _run(_context(cache), RecordingNext(_chunk("Hel"), ChatResponse({"error": "model crashed"})))
        assert cache.stored == []

    async def test_a_blank_answer_is_not_stored(self) -> None:
        cache = FakeSessionCache()
        await _run(_context(cache), RecordingNext(_chunk("  \n"), _chunk("", done=True)))
        assert cache.stored == []

    async def test_an_answer_without_done_is_not_stored(self) -> None:
        cache = FakeSessionCache()
        await _run(_context(cache), RecordingNext(_chunk("Hel"), _chunk("lo")))
        assert cache.stored == []


class TestInterrupted:
    @staticmethod
    def _endless(closed: asyncio.Event) -> RecordingNext:
        """A chain streaming one chunk, then waiting forever; `closed` is set once it is closed."""

        class Endless(RecordingNext):
            async def __call__(self, context: ChatContext) -> AsyncGenerator[ChatResponse]:
                self.calls.append(context)
                try:
                    yield _chunk("Hel")
                    await asyncio.sleep(3600)
                    yield _chunk("lo", done=True)
                finally:
                    closed.set()

        return Endless()

    async def test_closing_stores_nothing_and_closes_the_chain(self) -> None:
        cache = FakeSessionCache()
        closed = asyncio.Event()
        async with aclosing(CacheStage()(_context(cache), self._endless(closed))) as chunks:
            assert _contents([await anext(chunks)]) == ["Hel"]

        assert closed.is_set()
        assert cache.stored == []

    async def test_cancellation_propagates_and_stores_nothing(self) -> None:
        cache = FakeSessionCache()
        closed = asyncio.Event()
        started = asyncio.Event()

        async def consume() -> None:
            async with aclosing(CacheStage()(_context(cache), self._endless(closed))) as chunks:
                async for _ in chunks:
                    started.set()

        task = asyncio.create_task(consume())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert closed.is_set()
        assert cache.stored == []


class TestFailures:
    @pytest.mark.parametrize(
        "error", [CacheError("the Redis server does not answer"), EmbeddingError("no embedding model")]
    )
    async def test_a_failed_lookup_still_answers_from_the_chain(self, error: Exception) -> None:
        cache = FakeSessionCache(lookup_error=error)
        context = _context(cache)
        chunks = (_chunk("Hello"), _chunk("", done=True))

        assert await _run(context, RecordingNext(*chunks)) == list(chunks)

        [message] = _system_messages(context)
        assert message.startswith("The semantic cache could not be read:")

    async def test_a_failed_store_still_answers_and_is_reported_after(self) -> None:
        cache = FakeSessionCache(failing_store=True)
        context = _context(cache)
        chunks = (_chunk("Hello"), answer_chunk(""))

        assert await _run(context, RecordingNext(*chunks)) == list(chunks)

        [message] = _system_messages(context)
        assert message.startswith("The semantic cache could not be written:")
