import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing

import pytest
from sensai_server.llm.responses import ChatResponse
from sensai_server.pipeline import ChatContext, ChatPipeline, Next, Stage, answer_chunk

from tests.server.fakes import RecordingOutbound


def _context() -> ChatContext:
    return ChatContext(message="hi", out=RecordingOutbound(), model="m")


def _model(log: list[str], *, block: asyncio.Event | None = None) -> Next:
    async def model(_context: ChatContext) -> AsyncIterator[ChatResponse]:
        log.append("model")
        yield answer_chunk("a")
        if block is not None:
            await block.wait()
        yield answer_chunk("b")

    return model


def _stage(name: str, log: list[str]) -> Stage:
    async def stage(context: ChatContext, next_stage: Next) -> AsyncIterator[ChatResponse]:
        log.append(f"{name} in")
        try:
            async with aclosing(next_stage(context)) as chunks:
                async for chunk in chunks:
                    log.append(f"{name} chunk")
                    yield chunk
        finally:
            log.append(f"{name} closed")

    return stage


async def _contents(chunks: AsyncIterator[ChatResponse]) -> list[str | None]:
    async with aclosing(chunks) as stream:
        return [chunk.content async for chunk in stream]


class TestChatPipeline:
    async def test_no_stages_streams_model(self) -> None:
        log: list[str] = []
        pipeline = ChatPipeline(_model(log))
        assert await _contents(pipeline.execute(_context())) == ["a", "b"]
        assert log == ["model"]

    async def test_stage_order(self) -> None:
        log: list[str] = []
        pipeline = ChatPipeline(_model(log)).use(_stage("a", log)).use(_stage("b", log))
        assert await _contents(pipeline.execute(_context())) == ["a", "b"]
        assert log == [
            "a in",
            "b in",
            "model",
            "b chunk",
            "a chunk",
            "b chunk",
            "a chunk",
            "b closed",
            "a closed",
        ]

    def test_use_is_chainable(self) -> None:
        pipeline = ChatPipeline(_model([]))
        assert pipeline.use(_stage("a", [])) is pipeline

    async def test_execute_is_lazy(self) -> None:
        log: list[str] = []
        pipeline = ChatPipeline(_model(log)).use(_stage("a", log))
        chunks = pipeline.execute(_context())
        assert log == []
        await _contents(chunks)

    async def test_stage_answers_in_place_of_model(self) -> None:
        log: list[str] = []

        async def guard(_context: ChatContext, _next_stage: Next) -> AsyncIterator[ChatResponse]:
            yield answer_chunk("stop")

        pipeline = ChatPipeline(_model(log)).use(guard).use(_stage("a", log))
        assert await _contents(pipeline.execute(_context())) == ["stop"]
        assert log == []

    async def test_closing_closes_every_stage(self) -> None:
        log: list[str] = []
        pipeline = ChatPipeline(_model(log)).use(_stage("a", log)).use(_stage("b", log))
        async with aclosing(pipeline.execute(_context())) as chunks:
            async for _ in chunks:
                break
        assert log[-2:] == ["b closed", "a closed"]

    async def test_cancel_closes_every_stage(self) -> None:
        log: list[str] = []
        block = asyncio.Event()
        pipeline = ChatPipeline(_model(log, block=block)).use(_stage("a", log)).use(_stage("b", log))
        first = asyncio.Event()

        async def consume() -> None:
            async with aclosing(pipeline.execute(_context())) as chunks:
                async for _ in chunks:
                    first.set()

        task = asyncio.create_task(consume())
        await first.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert log[-2:] == ["b closed", "a closed"]
