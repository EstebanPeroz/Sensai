from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.http.exceptions import ApiException
from qdrant_client.models import Distance, VectorParams

from sensai.rag.collection import (
    DEFAULT_COLLECTION,
    Collection,
    as_chunk,
    by_source,
    chunk_point,
    open_client,
)
from sensai.rag.error import RagError, VectorStoreError

ENGINES = [1.0, 0.0]
CATS = [0.0, 1.0]


def point(source: str, position: int, text: str, vector: list[float]) -> object:
    """A chunk point, named the way the tests read."""
    return chunk_point(source, position, text, vector)


class TestOpenClient:
    def test_a_memory_location_is_passed_through(self) -> None:
        client = open_client(":memory:")

        assert client.init_options["location"] == ":memory:"
        client.close()

    def test_a_directory_is_opened_as_an_embedded_database(self, tmp_path: Path) -> None:
        location = str(tmp_path / "qdrant")

        client = open_client(location)

        assert client.init_options["path"] == location
        client.close()

    @pytest.mark.parametrize("location", ["http://localhost:6333", "https://qdrant.example:6333"])
    def test_a_url_is_opened_as_a_server(self, location: str) -> None:
        with patch("sensai.rag.collection.QdrantClient", return_value=MagicMock()) as client:
            open_client(location)

        assert client.call_args == ((location,), {})


class TestOpenCollection:
    def test_it_is_not_created_before_the_first_write(self, collection: Collection) -> None:
        assert not collection.exists()

    def test_it_has_no_dimension_before_the_first_write(self, collection: Collection) -> None:
        assert collection.dimension() is None

    def test_the_default_name_is_the_one_used(self, collection: Collection) -> None:
        assert collection.name == DEFAULT_COLLECTION

    def test_a_url_is_treated_as_served(self) -> None:
        with patch("sensai.rag.collection.QdrantClient", return_value=MagicMock()):
            opened = Collection.open("http://localhost:6333")

        assert opened._served

    def test_a_directory_is_not_treated_as_served(self, tmp_path: Path) -> None:
        opened = Collection.open(str(tmp_path / "qdrant"))

        assert not opened._served
        opened.close()

    def test_what_is_written_survives_a_new_client(self, collection: Collection, store_location: str) -> None:
        collection.upsert([point("a.md", 0, "x", ENGINES)])
        collection.close()

        reopened = Collection.open(store_location)

        assert reopened.count() == 1
        reopened.close()

    def test_two_collections_on_one_location_stay_apart(self, store_location: str) -> None:
        first = Collection.open(store_location, "one")
        first.upsert([point("a.md", 0, "x", ENGINES)])
        first.close()

        second = Collection.open(store_location, "two")

        assert second.count() == 0
        second.close()

    def test_a_qdrant_failure_when_checking_existence_is_a_vector_store_error(self) -> None:
        client = MagicMock()
        client.collection_exists.side_effect = ApiException("qdrant is unavailable")
        collection = Collection(client)

        with pytest.raises(VectorStoreError, match="could not check whether the collection"):
            collection.exists()


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
        assert filled.query(ENGINES)[0].payload["text"] == "engines"

    def test_every_hit_carries_its_score(self, filled: Collection) -> None:
        assert filled.query(ENGINES)[0].score == pytest.approx(1.0)

    def test_every_hit_carries_the_stored_payload(self, filled: Collection) -> None:
        payload = filled.query(ENGINES)[0].payload

        assert payload["source"] == "engines.md"
        assert payload["position"] == 0
        assert payload["text"] == "engines"

    def test_the_limit_caps_the_hits(self, filled: Collection) -> None:
        assert len(filled.query(ENGINES, limit=1)) == 1

    def test_a_minimum_score_drops_the_far_ones(self, filled: Collection) -> None:
        found = filled.query(ENGINES, min_score=0.5)

        assert [hit.payload["text"] for hit in found] == ["engines"]

    def test_a_minimum_score_can_drop_everything(self, filled: Collection) -> None:
        assert filled.query(ENGINES, min_score=1.1) == []

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
        collection = Collection(client)

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
        payload = point("a.md", 3, "some text", ENGINES).payload

        assert payload["source"] == "a.md"
        assert payload["position"] == 3
        assert payload["text"] == "some text"


class TestAsChunk:
    def test_it_carries_the_payload_and_the_score(self, collection: Collection) -> None:
        collection.upsert([point("a.md", 0, "x", ENGINES)])

        chunk = as_chunk(collection.query(ENGINES)[0])

        assert chunk["source"] == "a.md"
        assert chunk["text"] == "x"
        assert chunk["score"] == pytest.approx(1.0)
