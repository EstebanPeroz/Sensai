import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from qdrant_client import QdrantClient
from sensai_server.llm.error import EmbeddingError
from sensai_server.memory.persistent_db.error import InvalidInstanceError
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai_server.rag import Rag, RagError
from sensai_server.rag.collection import Collection
from sensai_server.rag.error import VectorStoreError

from tests.server.conftest import MODEL, db_url
from tests.server.fakes import FakeEmbedder

ENGINES = "# Engines\n\nThe engine burns fuel."
CATS = "# Cats\n\nCats sleep all day."


class WideEmbedder(FakeEmbedder):
    """A `FakeEmbedder` whose vectors have one more dimension, as if the embedding model had changed."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[*vector, 0.0] for vector in super().embed(texts)]


def client_of(rag: Rag) -> QdrantClient:
    """The Qdrant client `rag` opened."""
    return rag._client  # noqa: SLF001


@contextmanager
def opened(embedder: FakeEmbedder, db: SqlAlchemy) -> Iterator[Rag]:
    """A `Rag` on a fresh in-memory Qdrant, closed on exit."""
    rag = Rag(":memory:", embedder, db)
    try:
        yield rag
    finally:
        client_of(rag).close()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def db(tmp_path: Path) -> SqlAlchemy:
    return SqlAlchemy(db_url(tmp_path))


@pytest.fixture
def rag(embedder: FakeEmbedder, db: SqlAlchemy) -> Iterator[Rag]:
    with opened(embedder, db) as built:
        yield built


@pytest.fixture
def client(rag: Rag) -> QdrantClient:
    """The Qdrant of `rag`."""
    return client_of(rag)


def new_branch(db: SqlAlchemy) -> UUID:
    branch = db.create_branch(model=MODEL)
    assert branch is not None
    return branch


def texts(client: QdrantClient, rag_id: UUID) -> list[str]:
    """The texts stored in the collection of `rag_id`."""
    return sorted(payload["text"] for payload in Collection(client, rag_id).payloads())


class TestIndex:
    async def test_it_stores_the_chunks_in_the_collection_of_the_rag_id(self, rag: Rag, client: QdrantClient) -> None:
        rag_id = uuid4()

        count = await rag.index(rag_id, "engines.md", ENGINES)

        assert count == 1
        assert texts(client, rag_id) == ["Engines\n\nThe engine burns fuel."]

    async def test_another_rag_id_sees_nothing(self, rag: Rag, client: QdrantClient) -> None:
        await rag.index(uuid4(), "engines.md", ENGINES)

        assert texts(client, uuid4()) == []

    async def test_re_indexing_a_source_replaces_it(self, rag: Rag, client: QdrantClient) -> None:
        rag_id = uuid4()
        await rag.index(rag_id, "doc.md", "# A\n\nFirst.\n\n# B\n\nSecond.")

        await rag.index(rag_id, "doc.md", CATS)

        assert texts(client, rag_id) == ["Cats\n\nCats sleep all day."]

    async def test_other_sources_are_kept(self, rag: Rag, client: QdrantClient) -> None:
        rag_id = uuid4()
        await rag.index(rag_id, "engines.md", ENGINES)

        await rag.index(rag_id, "cats.md", CATS)

        assert len(texts(client, rag_id)) == 2

    async def test_stores_the_chunks_in_order_with_their_source(self, rag: Rag, client: QdrantClient) -> None:
        rag_id = uuid4()

        count = await rag.index(rag_id, "doc.md", "# A\n\nFirst.\n\n# B\n\nSecond.\n\n# C\n\nThird.")

        stored = sorted(Collection(client, rag_id).payloads(), key=lambda payload: payload["position"])
        assert count == 3
        assert [(p["source"], p["position"], p["text"]) for p in stored] == [
            ("doc.md", 0, "A\n\nFirst."),
            ("doc.md", 1, "B\n\nSecond."),
            ("doc.md", 2, "C\n\nThird."),
        ]

    async def test_stores_the_vectors_the_embedder_produced(
        self, rag: Rag, client: QdrantClient, embedder: FakeEmbedder
    ) -> None:
        rag_id = uuid4()

        await rag.index(rag_id, "engines.md", ENGINES)

        [point] = client.scroll(Collection(client, rag_id).name, with_vectors=True)[0]
        [expected] = embedder.embed(["Engines\n\nThe engine burns fuel."])
        norm = math.hypot(*expected)
        assert point.vector == pytest.approx([value / norm for value in expected])

    async def test_embeds_the_chunks_of_a_small_document_in_one_call(self, rag: Rag, embedder: FakeEmbedder) -> None:
        await rag.index(uuid4(), "doc.md", "# A\n\nFirst.\n\n# B\n\nSecond.")

        assert embedder.batches == [["A\n\nFirst.", "B\n\nSecond."]]

    async def test_re_indexing_an_unchanged_source_does_not_duplicate_it(self, rag: Rag, client: QdrantClient) -> None:
        rag_id = uuid4()
        await rag.index(rag_id, "engines.md", ENGINES)

        await rag.index(rag_id, "engines.md", ENGINES)

        assert Collection(client, rag_id).count() == 1

    async def test_a_vector_of_another_size_is_refused_and_keeps_what_is_stored(
        self, rag: Rag, client: QdrantClient, db: SqlAlchemy
    ) -> None:
        rag_id = uuid4()
        await rag.index(rag_id, "engines.md", ENGINES)

        with opened(WideEmbedder(), db) as wide:
            own, wide._client = wide._client, client  # noqa: SLF001 - the same Qdrant, embedded by another model
            try:
                with pytest.raises(VectorStoreError):
                    await wide.index(rag_id, "engines.md", CATS)
            finally:
                wide._client = own  # noqa: SLF001

        assert texts(client, rag_id) == ["Engines\n\nThe engine burns fuel."]

    async def test_a_text_holding_nothing_is_refused(self, rag: Rag, client: QdrantClient) -> None:
        rag_id = uuid4()

        with pytest.raises(RagError, match=r"no text could be extracted from empty\.md"):
            await rag.index(rag_id, "empty.md", "  \n<p></p>\n")
        assert not Collection(client, rag_id).exists()

    async def test_a_failing_embedder_stores_nothing(self, db: SqlAlchemy) -> None:
        with opened(FakeEmbedder(failing=True), db) as rag:
            rag_id = uuid4()

            with pytest.raises(EmbeddingError):
                await rag.index(rag_id, "engines.md", ENGINES)
            assert not Collection(client_of(rag), rag_id).exists()


class TestResearch:
    async def test_it_finds_the_closest_chunks_with_their_score(self, rag: Rag) -> None:
        rag_id = uuid4()
        await rag.index(rag_id, "engines.md", ENGINES)
        await rag.index(rag_id, "cats.md", CATS)

        chunks = await rag.research(rag_id, "How does an engine work?")

        assert [chunk["source"] for chunk in chunks] == ["engines.md"]
        assert chunks[0]["score"] > 0.5

    async def test_a_rag_id_with_nothing_indexed_finds_nothing(self, rag: Rag) -> None:
        assert await rag.research(uuid4(), "engine?") == []

    async def test_it_searches_only_its_own_rag_id(self, rag: Rag) -> None:
        await rag.index(uuid4(), "engines.md", ENGINES)

        assert await rag.research(uuid4(), "engine?") == []

    async def test_a_failing_embedder_raises(self, db: SqlAlchemy) -> None:
        with opened(FakeEmbedder(failing=True), db) as rag, pytest.raises(EmbeddingError):
            await rag.research(uuid4(), "engine?")


class TestOpenClient:
    def test_a_memory_location_is_passed_through(self, embedder: FakeEmbedder, db: SqlAlchemy) -> None:
        with opened(embedder, db) as rag:
            assert client_of(rag).init_options["location"] == ":memory:"
            assert not rag._served  # noqa: SLF001

    def test_a_directory_is_opened_as_an_embedded_database(
        self, embedder: FakeEmbedder, db: SqlAlchemy, tmp_path: Path
    ) -> None:
        location = str(tmp_path / "qdrant")
        rag = Rag(location, embedder, db)

        assert client_of(rag).init_options["path"] == location
        assert not rag._served  # noqa: SLF001
        client_of(rag).close()

    @pytest.mark.parametrize("location", ["http://localhost:6333", "https://qdrant.example:6333"])
    def test_a_url_is_opened_as_a_server(self, embedder: FakeEmbedder, db: SqlAlchemy, location: str) -> None:
        with patch("sensai_server.rag.rag.QdrantClient", return_value=MagicMock()) as client:
            rag = Rag(location, embedder, db)

        assert client.call_args == ((location,), {})
        assert rag._served  # noqa: SLF001


class TestCreate:
    async def test_it_creates_an_empty_collection_at_the_embedder_s_size(
        self, rag: Rag, db: SqlAlchemy, client: QdrantClient
    ) -> None:
        branch, rag_id = new_branch(db), uuid4()
        db.set_branch_rag_id(branch, rag_id)

        await rag.create(branch, rag_id)

        collection = Collection(client, rag_id)
        assert collection.count() == 0
        assert collection.dimension() == 2

    async def test_the_branch_is_then_linked_to_it(self, rag: Rag, db: SqlAlchemy, client: QdrantClient) -> None:
        branch, rag_id = new_branch(db), uuid4()
        db.set_branch_rag_id(branch, rag_id)

        await rag.create(branch, rag_id)
        await rag.drop_branch(branch)

        assert not Collection(client, rag_id).exists()

    async def test_the_size_is_probed_once(self, rag: Rag, db: SqlAlchemy, embedder: FakeEmbedder) -> None:
        for _ in range(2):
            branch, rag_id = new_branch(db), uuid4()
            db.set_branch_rag_id(branch, rag_id)
            await rag.create(branch, rag_id)

        assert len(embedder.batches) == 1

    async def test_a_failing_embedder_creates_nothing(self, db: SqlAlchemy) -> None:
        branch, rag_id = new_branch(db), uuid4()
        with opened(FakeEmbedder(failing=True), db) as rag:
            with pytest.raises(EmbeddingError):
                await rag.create(branch, rag_id)
            assert not Collection(client_of(rag), rag_id).exists()


class TestRagIdFor:
    async def test_a_branch_with_no_rag_gets_a_new_one_linked_at_once(self, rag: Rag, db: SqlAlchemy) -> None:
        branch = new_branch(db)

        rag_id = await rag.rag_id_for(branch)

        assert db.get_branch_rag_id(branch) == rag_id

    async def test_it_gives_the_same_id_every_time(self, rag: Rag, db: SqlAlchemy) -> None:
        branch = new_branch(db)

        assert await rag.rag_id_for(branch) == await rag.rag_id_for(branch)

    async def test_it_reads_the_link_stored_in_the_database(self, embedder: FakeEmbedder, db: SqlAlchemy) -> None:
        branch = new_branch(db)
        rag_id = uuid4()
        db.set_branch_rag_id(branch, rag_id)

        with opened(embedder, db) as rag:
            assert await rag.rag_id_for(branch) == rag_id

    async def test_an_unknown_branch_raises(self, rag: Rag) -> None:
        with pytest.raises(InvalidInstanceError):
            await rag.rag_id_for(uuid4())


class TestLink:
    async def test_it_stores_the_link(self, rag: Rag, db: SqlAlchemy) -> None:
        branch, rag_id = new_branch(db), uuid4()

        await rag.link(branch, rag_id)

        assert db.get_branch_rag_id(branch) == rag_id
        assert await rag.rag_id_for(branch) == rag_id

    async def test_the_documents_indexed_before_the_link_stay_reachable(self, rag: Rag, db: SqlAlchemy) -> None:
        branch, rag_id = new_branch(db), uuid4()
        await rag.index(rag_id, "engines.md", ENGINES)

        await rag.link(branch, rag_id)

        assert await rag.research(await rag.rag_id_for(branch), "engine?")

    async def test_an_unknown_branch_raises(self, rag: Rag) -> None:
        with pytest.raises(InvalidInstanceError):
            await rag.link(uuid4(), uuid4())


class TestFork:
    async def test_the_child_gets_a_new_rag_holding_the_parent_s_points(
        self, rag: Rag, db: SqlAlchemy, client: QdrantClient
    ) -> None:
        parent, child = new_branch(db), new_branch(db)
        parent_id = await rag.rag_id_for(parent)
        await rag.index(parent_id, "engines.md", ENGINES)

        child_id = await rag.fork(parent, child)

        assert child_id != parent_id
        assert await rag.rag_id_for(child) == child_id
        assert texts(client, child_id) == texts(client, parent_id)

    async def test_it_embeds_nothing(self, rag: Rag, db: SqlAlchemy, embedder: FakeEmbedder) -> None:
        parent, child = new_branch(db), new_branch(db)
        await rag.index(await rag.rag_id_for(parent), "engines.md", ENGINES)
        embedder.batches.clear()

        await rag.fork(parent, child)

        assert embedder.batches == []

    async def test_indexing_in_the_child_leaves_the_parent_unchanged(
        self, rag: Rag, db: SqlAlchemy, client: QdrantClient
    ) -> None:
        parent, child = new_branch(db), new_branch(db)
        parent_id = await rag.rag_id_for(parent)
        await rag.index(parent_id, "engines.md", ENGINES)
        child_id = await rag.fork(parent, child)

        await rag.index(child_id, "cats.md", CATS)

        assert len(texts(client, child_id)) == 2
        assert texts(client, parent_id) == ["Engines\n\nThe engine burns fuel."]

    async def test_a_parent_with_no_rag_gives_the_child_an_empty_one(
        self, rag: Rag, db: SqlAlchemy, client: QdrantClient
    ) -> None:
        parent, child = new_branch(db), new_branch(db)

        child_id = await rag.fork(parent, child)

        assert db.get_branch_rag_id(child) == child_id
        assert db.get_branch_rag_id(parent) is None
        assert not Collection(client, child_id).exists()


class TestDropBranch:
    async def test_it_removes_the_points(self, rag: Rag, db: SqlAlchemy, client: QdrantClient) -> None:
        branch = new_branch(db)
        rag_id = await rag.rag_id_for(branch)
        await rag.index(rag_id, "engines.md", ENGINES)

        await rag.drop_branch(branch)

        assert not Collection(client, rag_id).exists()

    async def test_it_leaves_the_other_branches_alone(self, rag: Rag, db: SqlAlchemy, client: QdrantClient) -> None:
        kept, dropped = new_branch(db), new_branch(db)
        kept_id = await rag.rag_id_for(kept)
        await rag.index(kept_id, "engines.md", ENGINES)
        await rag.index(await rag.rag_id_for(dropped), "engines.md", ENGINES)

        await rag.drop_branch(dropped)

        assert Collection(client, kept_id).count() == 1

    async def test_a_branch_with_no_rag_is_left_alone(self, rag: Rag, db: SqlAlchemy) -> None:
        branch = new_branch(db)

        await rag.drop_branch(branch)

        assert db.get_branch_rag_id(branch) is None

    async def test_the_branch_can_then_be_removed(self, rag: Rag, db: SqlAlchemy) -> None:
        branch = new_branch(db)
        await rag.index(await rag.rag_id_for(branch), "engines.md", ENGINES)

        await rag.drop_branch(branch)
        db.remove_branch(branch)

        assert branch not in db.get_branches()


class TestRelease:
    async def test_it_drops_an_unlinked_rag(self, rag: Rag, client: QdrantClient) -> None:
        rag_id = uuid4()
        await rag.index(rag_id, "engines.md", ENGINES)

        await rag.release(rag_id)

        assert not Collection(client, rag_id).exists()

    async def test_it_refuses_a_linked_rag(self, rag: Rag, db: SqlAlchemy, client: QdrantClient) -> None:
        branch, rag_id = new_branch(db), uuid4()
        await rag.index(rag_id, "engines.md", ENGINES)
        await rag.link(branch, rag_id)

        await rag.release(rag_id)

        assert Collection(client, rag_id).count() == 1

    async def test_it_refuses_a_rag_read_back_from_the_database(
        self, rag: Rag, db: SqlAlchemy, client: QdrantClient
    ) -> None:
        branch, rag_id = new_branch(db), uuid4()
        db.set_branch_rag_id(branch, rag_id)
        await rag.index(rag_id, "engines.md", ENGINES)
        await rag.rag_id_for(branch)

        await rag.release(rag_id)

        assert Collection(client, rag_id).count() == 1

    async def test_a_rag_nothing_was_indexed_in_is_left_alone(self, rag: Rag) -> None:
        await rag.release(uuid4())


class TestSources:
    def test_each_source_is_named_once_best_match_first(self) -> None:
        chunks = [{"source": "b.md"}, {"source": "a.md"}, {"source": "b.md"}]

        assert Rag.sources(chunks) == ["b.md", "a.md"]

    def test_no_chunk_names_no_source(self) -> None:
        assert Rag.sources([]) == []
