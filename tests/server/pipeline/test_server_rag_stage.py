from collections.abc import AsyncGenerator, Iterator
from contextlib import aclosing
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sensai_protocol import System
from sensai_server.llm.responses import ChatResponse
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai_server.pipeline import ChatContext
from sensai_server.pipeline.modules import RagStage
from sensai_server.pipeline.modules.rag import EXTRACTS_INSTRUCTION
from sensai_server.rag import Rag

from tests.server.conftest import db_url
from tests.server.fakes import FakeEmbedder, RecordingNext, RecordingOutbound, chunk

HISTORY = [{"role": "user", "content": "how does the engine work?"}]
QUESTION = "how does the engine work?"
SOURCE = "engines.md"
ENGINES = "# Engines\n\nThe engine burns fuel."


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def rag(embedder: FakeEmbedder, tmp_path: Path) -> Iterator[Rag]:
    built = Rag(":memory:", embedder, SqlAlchemy(db_url(tmp_path)))
    try:
        yield built
    finally:
        built._client.close()  # noqa: SLF001


@pytest.fixture
async def rag_id(rag: Rag) -> UUID:
    """A RAG id holding the engines document."""
    rag_id = uuid4()
    await rag.index(rag_id, SOURCE, ENGINES)
    return rag_id


def _context(rag_id: UUID | None, out: RecordingOutbound | None = None) -> ChatContext:
    return ChatContext(
        message=QUESTION,
        out=out or RecordingOutbound(),
        model="m",
        messages=[dict(message) for message in HISTORY],
        rag_id=rag_id,
    )


def _answer() -> RecordingNext:
    return RecordingNext(ChatResponse(chunk("It ")), ChatResponse(chunk("burns fuel.", done=True)))


async def _run(rag: Rag, context: ChatContext, next_stage: RecordingNext) -> list[ChatResponse]:
    async with aclosing(RagStage(rag)(context, next_stage)) as chunks:
        return [item async for item in chunks]


class TestRagStage:
    async def test_extracts_after_history(self, rag: Rag, rag_id: UUID) -> None:
        next_stage = _answer()
        await _run(rag, _context(rag_id), next_stage)
        [messages] = next_stage.seen_messages
        assert messages[:-1] == HISTORY
        extracts = messages[-1]
        assert extracts["role"] == "system"
        assert extracts["content"].startswith(EXTRACTS_INSTRUCTION)
        assert f"[1] {SOURCE} (passage 0)" in extracts["content"]
        assert "The engine burns fuel." in extracts["content"]

    async def test_sources_before_single_done(self, rag: Rag, rag_id: UUID) -> None:
        chunks = await _run(rag, _context(rag_id), _answer())
        assert [item.content for item in chunks] == ["It ", f"\n\nSources: {SOURCE}", "burns fuel."]
        assert [item.done for item in chunks] == [False, False, True]

    async def test_no_rag_id_leaves_payload(self, rag: Rag, embedder: FakeEmbedder) -> None:
        next_stage = _answer()
        chunks = await _run(rag, _context(None), next_stage)
        assert next_stage.seen_messages == [HISTORY]
        assert [item.content for item in chunks] == ["It ", "burns fuel."]
        assert embedder.batches == []

    async def test_nothing_found_leaves_payload(self, rag: Rag) -> None:
        next_stage = _answer()
        chunks = await _run(rag, _context(uuid4()), next_stage)
        assert next_stage.seen_messages == [HISTORY]
        assert [item.content for item in chunks] == ["It ", "burns fuel."]

    async def test_failing_search_still_answers(self, rag: Rag, rag_id: UUID, embedder: FakeEmbedder) -> None:
        embedder.failing = True
        out = RecordingOutbound()
        next_stage = _answer()
        chunks = await _run(rag, _context(rag_id, out), next_stage)
        assert next_stage.seen_messages == [HISTORY]
        assert [item.content for item in chunks] == ["It ", "burns fuel."]
        [message] = out.sent
        assert isinstance(message, System)
        assert message.message.startswith("The indexed documents could not be searched: ")

    async def test_no_sources_on_error(self, rag: Rag, rag_id: UUID) -> None:
        failed = ChatResponse({"done": True, "error": "model crashed"})
        chunks = await _run(rag, _context(rag_id), RecordingNext(failed))
        assert chunks == [failed]

    async def test_closing_closes_model_stream(self, rag: Rag, rag_id: UUID) -> None:
        closed = False

        async def endless(_context: ChatContext) -> AsyncGenerator[ChatResponse]:
            nonlocal closed
            try:
                while True:
                    yield ChatResponse(chunk("a"))
            finally:
                closed = True

        async with aclosing(RagStage(rag)(_context(rag_id), endless)) as chunks:
            async for _ in chunks:
                break
        assert closed
