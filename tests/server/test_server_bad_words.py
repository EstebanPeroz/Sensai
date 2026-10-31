from contextlib import aclosing

import pytest
from sensai_server.pipeline import ChatContext, answer_chunk
from sensai_server.pipeline.modules.guardrails import WARNING, BadWordsGuard

from tests.server.fakes import RecordingNext, RecordingOutbound


def _context(message: str) -> ChatContext:
    return ChatContext(message=message, out=RecordingOutbound(), model="m")


async def _run(guard: BadWordsGuard, message: str, next_stage: RecordingNext) -> list:
    async with aclosing(guard(_context(message), next_stage)) as chunks:
        return [chunk async for chunk in chunks]


class TestBadWordsGuard:
    async def test_abusive_answers_warning(self) -> None:
        next_stage = RecordingNext(answer_chunk("model"))
        chunks = await _run(BadWordsGuard(), "you are an asshole", next_stage)
        assert len(chunks) == 1
        assert chunks[0].content == WARNING
        assert chunks[0].done
        assert next_stage.calls == []

    async def test_clean_is_relayed(self) -> None:
        first, last = answer_chunk("a"), answer_chunk("b")
        next_stage = RecordingNext(first, last)
        assert await _run(BadWordsGuard(), "hello", next_stage) == [first, last]
        assert len(next_stage.calls) == 1

    @pytest.mark.parametrize("message", ["FUCK", "Fucking hell", "you BITCH!"])
    async def test_ignores_case(self, message: str) -> None:
        next_stage = RecordingNext()
        chunks = await _run(BadWordsGuard(), message, next_stage)
        assert [chunk.content for chunk in chunks] == [WARNING]
        assert next_stage.calls == []

    @pytest.mark.parametrize("message", ["classic", "assholes", "scunthorpe", "bitchy"])
    async def test_matches_whole_words(self, message: str) -> None:
        next_stage = RecordingNext(answer_chunk("ok"))
        chunks = await _run(BadWordsGuard(), message, next_stage)
        assert [chunk.content for chunk in chunks] == ["ok"]

    async def test_custom_words(self) -> None:
        guard = BadWordsGuard(frozenset({"darn"}))
        assert [c.content for c in await _run(guard, "darn it", RecordingNext())] == [WARNING]
        assert [c.content for c in await _run(guard, "fuck", RecordingNext(answer_chunk("ok")))] == ["ok"]
