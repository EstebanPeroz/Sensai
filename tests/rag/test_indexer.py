from pathlib import Path

import pytest

from sensai.rag.error import RagError, SourceError, SourceNotFoundError, UnsupportedFormatError
from sensai.rag.indexation_rag import ChunkStrategy
from sensai.rag.indexer import Indexer
from sensai.rag.qdrant_store import QdrantStore
from tests.rag.fakes import FakeIndexation

CATS_THEN_ENGINES = (
    "The cat sleeps all day. It purrs when you pet it. Cats sleep sixteen hours. "
    "The engine burns fuel. The engine efficiency is low. The engine wastes heat."
)


@pytest.fixture
def indexer(indexation: FakeIndexation, store: QdrantStore) -> Indexer:
    """An Indexer wired to a fake pipeline and an in-memory store."""
    return Indexer(indexation, store)


def stored(store: QdrantStore) -> list[dict]:
    """Every stored payload, ordered by source then position."""
    return sorted(store._payloads(None), key=lambda payload: (payload["source"], payload["position"]))


def write(tmp_path: Path, name: str, content: str) -> str:
    """Write `content` to `name` under `tmp_path` and return its path."""
    source = tmp_path / name
    source.write_text(content, encoding="utf-8")
    return str(source)


class TestIndex:
    def test_returns_the_number_of_stored_chunks(self, indexer: Indexer, tmp_path: Path) -> None:
        source = write(tmp_path, "a.txt", CATS_THEN_ENGINES)

        assert indexer.index(source) == 2

    def test_stores_the_chunks_under_their_source(self, indexer: Indexer, store: QdrantStore, tmp_path: Path) -> None:
        source = write(tmp_path, "a.txt", CATS_THEN_ENGINES)

        indexer.index(source)

        assert [payload["source"] for payload in stored(store)] == [source, source]

    def test_stores_the_chunks_in_order(self, indexer: Indexer, store: QdrantStore, tmp_path: Path) -> None:
        source = write(tmp_path, "a.txt", CATS_THEN_ENGINES)

        indexer.index(source)

        chunks = stored(store)
        assert [payload["position"] for payload in chunks] == [0, 1]
        assert chunks[0]["text"].startswith("The cat sleeps")
        assert chunks[1]["text"].startswith("The engine burns")

    def test_stores_the_model_the_pipeline_reports(
        self, indexer: Indexer, indexation: FakeIndexation, store: QdrantStore, tmp_path: Path
    ) -> None:
        source = write(tmp_path, "a.txt", "One. Two.")

        indexer.index(source)

        assert {payload["model"] for payload in stored(store)} == {indexation.model}

    def test_stores_the_vectors_the_pipeline_produced(
        self, indexer: Indexer, store: QdrantStore, tmp_path: Path
    ) -> None:
        source = write(tmp_path, "a.txt", "The cat sleeps.")

        indexer.index(source)

        assert store.dimension() == 2
        assert store.count() == 1

    def test_embeds_every_chunk_in_one_call(self, indexer: Indexer, indexation: FakeIndexation, tmp_path: Path) -> None:
        source = write(tmp_path, "a.txt", CATS_THEN_ENGINES)

        indexer.index(source)

        assert len(indexation.batches[-1]) == 2

    def test_a_markdown_source_carries_its_titles(self, indexer: Indexer, store: QdrantStore, tmp_path: Path) -> None:
        source = write(tmp_path, "a.md", "# Engines\nThe engine burns fuel.")

        indexer.index(source)

        assert stored(store)[0]["text"] == "Engines\n\nThe engine burns fuel."

    def test_a_forced_strategy_is_passed_through(self, indexer: Indexer, store: QdrantStore, tmp_path: Path) -> None:
        source = write(tmp_path, "a.txt", CATS_THEN_ENGINES)

        indexer.index(source, strategy=ChunkStrategy.SENTENCE)

        assert len(stored(store)) == 1


class TestIndexAgain:
    def test_replaces_what_was_stored_for_the_source(
        self, indexer: Indexer, store: QdrantStore, tmp_path: Path
    ) -> None:
        source = write(tmp_path, "a.txt", CATS_THEN_ENGINES)
        indexer.index(source)

        Path(source).write_text("The cat sleeps.", encoding="utf-8")
        assert indexer.index(source) == 1

        assert [payload["text"] for payload in stored(store)] == ["The cat sleeps."]

    def test_does_not_duplicate_an_unchanged_source(self, indexer: Indexer, store: QdrantStore, tmp_path: Path) -> None:
        source = write(tmp_path, "a.txt", CATS_THEN_ENGINES)

        indexer.index(source)
        indexer.index(source)

        assert len(stored(store)) == 2

    def test_leaves_the_other_sources_alone(self, indexer: Indexer, store: QdrantStore, tmp_path: Path) -> None:
        first = write(tmp_path, "a.txt", "The cat sleeps.")
        second = write(tmp_path, "b.txt", "The engine burns fuel.")
        indexer.index(first)
        indexer.index(second)

        indexer.index(first)

        assert {payload["source"] for payload in stored(store)} == {first, second}


class TestIndexRejects:
    def test_a_source_holding_no_text(self, indexer: Indexer, tmp_path: Path) -> None:
        source = write(tmp_path, "a.txt", "   ")

        with pytest.raises(RagError, match="no text could be extracted"):
            indexer.index(source)

    def test_a_missing_source(self, indexer: Indexer, tmp_path: Path) -> None:
        with pytest.raises(SourceNotFoundError):
            indexer.index(str(tmp_path / "absent.md"))

    def test_an_unsupported_format(self, indexer: Indexer, tmp_path: Path) -> None:
        source = write(tmp_path, "a.docx", "data")

        with pytest.raises(UnsupportedFormatError):
            indexer.index(source)

    def test_every_failure_is_a_rag_error(self, indexer: Indexer, tmp_path: Path) -> None:
        with pytest.raises(RagError):
            indexer.index(str(tmp_path / "absent.md"))

    def test_a_collection_failure_is_a_source_error(self, indexer: Indexer, tmp_path: Path) -> None:
        with pytest.raises(SourceError):
            indexer.index(str(tmp_path / "absent.md"))

    def test_an_empty_document_is_not_a_source_error(self, indexer: Indexer, tmp_path: Path) -> None:
        source = write(tmp_path, "a.txt", "")

        with pytest.raises(RagError) as caught:
            indexer.index(source)

        assert not isinstance(caught.value, SourceError)

    def test_stores_nothing_when_it_rejects(self, indexer: Indexer, store: QdrantStore, tmp_path: Path) -> None:
        with pytest.raises(RagError):
            indexer.index(str(tmp_path / "absent.md"))

        assert stored(store) == []
