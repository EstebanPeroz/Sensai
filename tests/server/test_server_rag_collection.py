from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
from sensai_server.rag.collection import Collection, as_chunk, by_source, chunk_point
from sensai_server.rag.error import RagError, VectorStoreError

ENGINES = [1.0, 0.0]
CATS = [0.0, 1.0]


def point(source: str, position: int, text: str, vector: list[float]) -> PointStruct:
    """A chunk point, named the way the tests read."""
    return chunk_point(source, position, text, vector)


@pytest.fixture
def client() -> Iterator[QdrantClient]:
    """A fresh in-memory Qdrant, shared by every collection of the test."""
    opened = QdrantClient(":memory:")
    yield opened
    opened.close()


@pytest.fixture
def rag_id() -> UUID:
    return uuid4()


@pytest.fixture
def collection(client: QdrantClient, rag_id: UUID) -> Collection:
    return Collection(client, rag_id)


class TestOpenCollection:
    def test_it_is_named_after_its_rag_id(self, collection: Collection, rag_id: UUID) -> None:
        assert collection.name == str(rag_id)

    def test_it_is_not_created_before_the_first_write(self, collection: Collection) -> None:
        assert not collection.exists()

    def test_it_has_no_dimension_before_the_first_write(self, collection: Collection) -> None:
        assert collection.dimension() is None

    def test_what_is_written_survives_a_new_client(self, tmp_path: Path, rag_id: UUID) -> None:
        location = str(tmp_path / "qdrant")
        first = QdrantClient(path=location)
        Collection(first, rag_id).upsert([point("a.md", 0, "x", ENGINES)])
        first.close()

        reopened = QdrantClient(path=location)

        assert Collection(reopened, rag_id).count() == 1
        reopened.close()

    def test_a_qdrant_failure_when_checking_existence_is_a_vector_store_error(self) -> None:
        client = MagicMock()
        client.collection_exists.side_effect = ApiException("qdrant is unavailable")
        collection = Collection(client, uuid4())

        with pytest.raises(VectorStoreError, match="could not check whether the collection"):
            collection.exists()


class TestIsolation:
    def test_two_rag_ids_never_see_each_other_s_points(self, client: QdrantClient) -> None:
        first, second = Collection(client, uuid4()), Collection(client, uuid4())
        first.upsert([point("same.md", 0, "first", ENGINES)])
        second.upsert([point("same.md", 0, "second", ENGINES)])

        assert [payload["text"] for payload in first.payloads()] == ["first"]
        assert [payload["text"] for payload in second.payloads()] == ["second"]

    def test_a_query_finds_only_its_own_collection_s_points(self, client: QdrantClient) -> None:
        first, second = Collection(client, uuid4()), Collection(client, uuid4())
        first.upsert([point("a.md", 0, "first", ENGINES)])
        second.upsert([point("b.md", 0, "second", ENGINES)])

        assert [hit.payload["text"] for hit in first.query(ENGINES) if hit.payload] == ["first"]

    def test_deleting_a_source_leaves_the_other_collection_alone(self, client: QdrantClient) -> None:
        first, second = Collection(client, uuid4()), Collection(client, uuid4())
        first.upsert([point("a.md", 0, "x", ENGINES)])
        second.upsert([point("a.md", 0, "x", ENGINES)])

        first.delete(by_source("a.md"))

        assert first.count() == 0
        assert second.count() == 1

    def test_one_collection_of_another_dimension_does_not_bother_the_other(self, client: QdrantClient) -> None:
        first, second = Collection(client, uuid4()), Collection(client, uuid4())
        first.upsert([point("a.md", 0, "x", ENGINES)])

        second.upsert([point("a.md", 0, "x", [1.0, 0.0, 0.0])])

        assert first.dimension() == 2
        assert second.dimension() == 3


class TestEnsure:
    def test_it_creates_the_collection_at_the_asked_dimension(self, collection: Collection) -> None:
        collection.ensure(2)

        assert collection.dimension() == 2

    def test_asking_again_for_the_same_dimension_changes_nothing(self, collection: Collection) -> None:
        collection.ensure(2)
        collection.upsert([point("a.md", 0, "x", ENGINES)])

        collection.ensure(2)

        assert collection.count() == 1

    def test_another_dimension_is_refused(self, collection: Collection) -> None:
        collection.ensure(2)

        with pytest.raises(VectorStoreError, match="holds 2-dimension vectors, not 3"):
            collection.ensure(3)

    def test_that_refusal_says_to_re_index(self, collection: Collection) -> None:
        collection.ensure(2)

        with pytest.raises(VectorStoreError, match="re-index every source"):
            collection.ensure(3)

    def test_that_refusal_is_a_rag_error(self, collection: Collection) -> None:
        collection.ensure(2)

        with pytest.raises(RagError):
            collection.ensure(3)

    def test_a_served_collection_indexes_the_source(self) -> None:
        client = MagicMock()
        client.collection_exists.return_value = False
        collection = Collection(client, uuid4(), served=True)

        collection.ensure(2)

        client.create_payload_index.assert_called_once_with(
            collection.name,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    def test_an_embedded_collection_does_not_index_the_source(self) -> None:
        client = MagicMock()
        client.collection_exists.return_value = False

        Collection(client, uuid4()).ensure(2)

        client.create_payload_index.assert_not_called()


class TestUpsert:
    def test_it_returns_how_many_points_were_written(self, collection: Collection) -> None:
        assert collection.upsert([point("a.md", 0, "x", ENGINES), point("a.md", 1, "y", CATS)]) == 2

    def test_it_creates_the_collection_on_the_first_write(self, collection: Collection) -> None:
        collection.upsert([point("a.md", 0, "x", ENGINES)])

        assert collection.dimension() == 2

    def test_no_point_writes_nothing(self, collection: Collection) -> None:
        assert collection.upsert([]) == 0
        assert not collection.exists()

    def test_no_point_leaves_what_is_stored_alone(self, collection: Collection) -> None:
        collection.upsert([point("a.md", 0, "x", ENGINES)])

        collection.upsert([])

        assert collection.count() == 1

    def test_a_point_written_twice_is_stored_once(self, collection: Collection) -> None:
        collection.upsert([point("a.md", 0, "first", ENGINES)])
        collection.upsert([point("a.md", 0, "second", ENGINES)])

        assert collection.count() == 1
        assert collection.payloads()[0]["text"] == "second"

    def test_mixed_dimensions_are_refused(self, collection: Collection) -> None:
        points = [point("a.md", 0, "x", ENGINES), point("a.md", 1, "y", [1.0, 0.0, 0.0])]

        with pytest.raises(VectorStoreError, match=r"inconsistent dimensions in the batch: \[2, 3\]"):
            collection.upsert(points)

    def test_a_dimension_the_collection_does_not_hold_is_refused(self, collection: Collection) -> None:
        collection.upsert([point("a.md", 0, "x", ENGINES)])

        with pytest.raises(VectorStoreError, match="holds 2-dimension vectors, not 3"):
            collection.upsert([point("b.md", 0, "y", [1.0, 0.0, 0.0])])

    def test_it_writes_nothing_when_it_rejects(self, collection: Collection) -> None:
        collection.upsert([point("a.md", 0, "x", ENGINES)])

        with pytest.raises(VectorStoreError):
            collection.upsert([point("b.md", 0, "y", [1.0, 0.0, 0.0])])

        assert collection.count() == 1


class TestQuery:
    @pytest.fixture
    def filled(self, collection: Collection) -> Collection:
        """A collection holding one point on each of the two axes."""
        collection.upsert([point("engines.md", 0, "engines", ENGINES), point("cats.md", 0, "cats", CATS)])
        return collection

    def test_the_closest_point_comes_first(self, filled: Collection) -> None:
        assert filled.query(ENGINES)[0].payload == {"source": "engines.md", "position": 0, "text": "engines"}

    def test_every_hit_carries_its_score(self, filled: Collection) -> None:
        assert filled.query(ENGINES)[0].score == pytest.approx(1.0)

    def test_the_limit_caps_the_hits(self, filled: Collection) -> None:
        assert len(filled.query(ENGINES, limit=1)) == 1

    def test_a_minimum_score_drops_the_far_ones(self, filled: Collection) -> None:
        found = filled.query(ENGINES, min_score=0.5)

        assert [hit.payload["text"] for hit in found if hit.payload] == ["engines"]

    def test_a_minimum_score_can_drop_everything(self, filled: Collection) -> None:
        assert filled.query(ENGINES, min_score=1.1) == []

    def test_a_filter_keeps_only_what_it_matches(self, filled: Collection) -> None:
        found = filled.query(ENGINES, where=by_source("cats.md"))

        assert [hit.payload["text"] for hit in found if hit.payload] == ["cats"]

    def test_querying_before_the_first_write_finds_nothing(self, collection: Collection) -> None:
        assert collection.query(ENGINES) == []

    def test_a_vector_of_the_wrong_size_is_refused(self, filled: Collection) -> None:
        with pytest.raises(VectorStoreError, match="3-dimension vector, but the collection holds 2"):
            filled.query([1.0, 0.0, 0.0])

    def test_a_qdrant_failure_when_querying_is_a_vector_store_error(self) -> None:
        client = MagicMock()
        client.collection_exists.return_value = True
        client.get_collection.return_value.config.params.vectors = VectorParams(size=2, distance=Distance.COSINE)
        client.query_points.side_effect = ApiException("qdrant is unavailable")
        collection = Collection(client, uuid4())

        with pytest.raises(VectorStoreError, match="could not search the collection"):
            collection.query(ENGINES)


class TestDelete:
    def test_it_drops_the_points_the_filter_matches(self, collection: Collection) -> None:
        collection.upsert([point("a.md", 0, "x", ENGINES), point("b.md", 0, "y", CATS)])

        collection.delete(by_source("a.md"))

        assert [payload["source"] for payload in collection.payloads()] == ["b.md"]

    def test_a_filter_matching_nothing_drops_nothing(self, collection: Collection) -> None:
        collection.upsert([point("a.md", 0, "x", ENGINES)])

        collection.delete(by_source("absent.md"))

        assert collection.count() == 1

    def test_it_does_nothing_before_the_first_write(self, collection: Collection) -> None:
        collection.delete(by_source("a.md"))

        assert not collection.exists()


class TestDrop:
    def test_it_removes_the_collection(self, collection: Collection) -> None:
        collection.upsert([point("a.md", 0, "x", ENGINES)])

        collection.drop()

        assert not collection.exists()
        assert collection.count() == 0

    def test_it_does_nothing_when_the_collection_does_not_exist(self, collection: Collection) -> None:
        collection.drop()

        assert not collection.exists()

    def test_it_leaves_the_other_collections_alone(self, client: QdrantClient, collection: Collection) -> None:
        other = Collection(client, uuid4())
        collection.upsert([point("a.md", 0, "x", ENGINES)])
        other.upsert([point("a.md", 0, "x", ENGINES)])

        collection.drop()

        assert other.count() == 1

    def test_a_qdrant_failure_is_a_vector_store_error(self) -> None:
        client = MagicMock()
        client.collection_exists.return_value = True
        client.delete_collection.side_effect = ApiException("qdrant is unavailable")

        with pytest.raises(VectorStoreError, match="could not delete the collection"):
            Collection(client, uuid4()).drop()


class TestCopyTo:
    @pytest.fixture
    def target(self, client: QdrantClient) -> Collection:
        return Collection(client, uuid4())

    def test_it_keeps_the_payloads(self, collection: Collection, target: Collection) -> None:
        collection.upsert([point("a.md", 0, "x", ENGINES), point("b.md", 3, "y", CATS)])

        collection.copy_to(target)

        assert sorted(target.payloads(), key=lambda payload: payload["source"]) == [
            {"source": "a.md", "position": 0, "text": "x"},
            {"source": "b.md", "position": 3, "text": "y"},
        ]

    def test_it_keeps_the_vectors(self, collection: Collection, target: Collection) -> None:
        collection.upsert([point("engines.md", 0, "engines", ENGINES), point("cats.md", 0, "cats", CATS)])

        collection.copy_to(target)

        hit = target.query(CATS)[0]
        assert hit.payload == {"source": "cats.md", "position": 0, "text": "cats"}
        assert hit.score == pytest.approx(1.0)

    def test_it_keeps_the_ids(self, collection: Collection, target: Collection) -> None:
        collection.upsert([point("a.md", 0, "x", ENGINES)])
        collection.copy_to(target)

        target.upsert([point("a.md", 0, "rewritten", ENGINES)])

        assert [payload["text"] for payload in target.payloads()] == ["rewritten"]

    def test_it_returns_how_many_points_were_copied(self, collection: Collection, target: Collection) -> None:
        collection.upsert([point("a.md", 0, "x", ENGINES), point("a.md", 1, "y", CATS)])

        assert collection.copy_to(target) == 2

    def test_it_pages_past_one_scroll(self, collection: Collection, target: Collection) -> None:
        collection.upsert([point("a.md", position, "x", ENGINES) for position in range(300)])

        collection.copy_to(target)

        assert target.count() == 300

    def test_it_leaves_the_source_unchanged(self, collection: Collection, target: Collection) -> None:
        collection.upsert([point("a.md", 0, "x", ENGINES)])

        collection.copy_to(target)
        target.upsert([point("b.md", 0, "y", CATS)])

        assert [payload["source"] for payload in collection.payloads()] == ["a.md"]

    def test_a_missing_source_creates_nothing(self, collection: Collection, target: Collection) -> None:
        assert collection.copy_to(target) == 0
        assert not target.exists()

    def test_an_empty_source_creates_nothing(self, collection: Collection, target: Collection) -> None:
        collection.ensure(2)

        assert collection.copy_to(target) == 0
        assert not target.exists()

    def test_a_target_of_another_dimension_is_refused(self, collection: Collection, target: Collection) -> None:
        collection.upsert([point("a.md", 0, "x", ENGINES)])
        target.ensure(3)

        with pytest.raises(VectorStoreError, match="holds 3-dimension vectors, not 2"):
            collection.copy_to(target)


class TestCount:
    def test_it_is_zero_before_the_first_write(self, collection: Collection) -> None:
        assert collection.count() == 0

    def test_it_counts_everything_by_default(self, collection: Collection) -> None:
        collection.upsert([point("a.md", 0, "x", ENGINES), point("b.md", 0, "y", CATS)])

        assert collection.count() == 2

    def test_it_counts_what_the_filter_matches(self, collection: Collection) -> None:
        collection.upsert([point("a.md", 0, "x", ENGINES), point("b.md", 0, "y", CATS)])

        assert collection.count(by_source("a.md")) == 1


class TestPayloads:
    def test_there_are_none_before_the_first_write(self, collection: Collection) -> None:
        assert collection.payloads() == []

    def test_it_pages_past_one_scroll(self, collection: Collection) -> None:
        collection.upsert([point("a.md", position, "x", ENGINES) for position in range(300)])

        assert len(collection.payloads()) == 300

    def test_it_reads_only_what_the_filter_matches(self, collection: Collection) -> None:
        collection.upsert([point("a.md", 0, "x", ENGINES), point("b.md", 0, "y", CATS)])

        assert [payload["source"] for payload in collection.payloads(by_source("b.md"))] == ["b.md"]


class TestChunkPoint:
    def test_the_same_source_and_position_give_the_same_id(self) -> None:
        assert point("a.md", 0, "x", ENGINES).id == point("a.md", 0, "other text", CATS).id

    def test_another_position_gives_another_id(self) -> None:
        assert point("a.md", 0, "x", ENGINES).id != point("a.md", 1, "x", ENGINES).id

    def test_another_source_gives_another_id(self) -> None:
        assert point("a.md", 0, "x", ENGINES).id != point("b.md", 0, "x", ENGINES).id

    def test_it_carries_the_source_position_and_text(self) -> None:
        assert point("a.md", 3, "some text", ENGINES).payload == {"source": "a.md", "position": 3, "text": "some text"}


class TestAsChunk:
    def test_it_carries_the_payload_and_the_score(self, collection: Collection) -> None:
        collection.upsert([point("a.md", 0, "x", ENGINES)])

        chunk = as_chunk(collection.query(ENGINES)[0])

        assert chunk["source"] == "a.md"
        assert chunk["text"] == "x"
        assert chunk["score"] == pytest.approx(1.0)
