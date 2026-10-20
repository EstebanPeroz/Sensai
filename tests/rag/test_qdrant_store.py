from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import PayloadSchemaType

from sensai.rag.error import RagError, VectorStoreError
from sensai.rag.qdrant_store import DEFAULT_COLLECTION, QdrantStore


def payloads(store: QdrantStore) -> list[dict]:
    """Every stored payload, ordered by source then position."""
    return sorted(store._payloads(None), key=lambda payload: (payload["source"], payload["position"]))


class TestInit:
    def test_starts_empty(self, store: QdrantStore) -> None:
        assert store.count() == 0

    def test_the_collection_is_not_created_before_the_first_write(self, store: QdrantStore) -> None:
        assert store.dimension() is None

    def test_the_first_write_creates_it_at_the_vectors_dimension(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["x"], [[0.1, 0.2, 0.3]], "m")

        assert store.dimension() == 3

    def test_data_survives_a_new_instance(self, store: QdrantStore, store_location: str) -> None:
        store.add_chunks("a.md", ["x"], [[0.1]], "m")
        store.close()

        reopened = QdrantStore(store_location)
        try:
            assert [payload["text"] for payload in payloads(reopened)] == ["x"]
        finally:
            reopened.close()

    def test_a_second_store_can_use_another_collection(self, store_location: str) -> None:
        first = QdrantStore(store_location, collection="one")
        first.add_chunks("a.md", ["in one"], [[0.1]], "m")
        first.close()

        second = QdrantStore(store_location, collection="two")
        try:
            assert second.count() == 0
        finally:
            second.close()

    def test_the_default_collection_is_the_one_used(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["x"], [[0.1]], "m")

        assert store._client.collection_exists(DEFAULT_COLLECTION)


class TestAddChunks:
    def test_returns_how_many_were_written(self, store: QdrantStore) -> None:
        assert store.add_chunks("a.md", ["x", "y"], [[0.1], [0.2]], "m") == 2

    def test_stores_the_text_source_model_and_position(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["hello"], [[0.1, 0.2, 0.3]], "bge-m3")

        stored = payloads(store)[0]
        assert stored["source"] == "a.md"
        assert stored["position"] == 0
        assert stored["text"] == "hello"
        assert stored["model"] == "bge-m3"

    def test_stamps_the_write(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["x"], [[0.1]], "m")

        assert payloads(store)[0]["created_at"]

    def test_positions_follow_the_list_order(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["first", "second", "third"], [[0.1], [0.2], [0.3]], "m")

        assert [(p["position"], p["text"]) for p in payloads(store)] == [
            (0, "first"),
            (1, "second"),
            (2, "third"),
        ]

    def test_every_point_gets_its_own_id(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["x", "y"], [[0.1], [0.2]], "m")
        store.add_chunks("b.md", ["z"], [[0.3]], "m")

        assert store.count() == 3

    def test_replaces_the_previous_chunks_of_the_source(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["old", "stale"], [[0.1], [0.2]], "m")

        store.add_chunks("a.md", ["fresh"], [[0.3]], "m")

        assert [payload["text"] for payload in payloads(store)] == ["fresh"]

    def test_leaves_the_other_sources_alone(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["from a"], [[0.1]], "m")
        store.add_chunks("b.md", ["from b"], [[0.2]], "m")

        store.add_chunks("a.md", ["new a"], [[0.3]], "m")

        assert [(p["source"], p["text"]) for p in payloads(store)] == [("a.md", "new a"), ("b.md", "from b")]

    def test_empty_list_writes_nothing(self, store: QdrantStore) -> None:
        assert store.add_chunks("a.md", [], [], "m") == 0
        assert store.count() == 0

    def test_empty_list_does_not_delete_what_is_stored(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["keep me"], [[0.1]], "m")

        store.add_chunks("a.md", [], [], "m")

        assert [payload["text"] for payload in payloads(store)] == ["keep me"]


class TestAddChunksWithoutReplace:
    def test_appends_instead_of_replacing(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["first"], [[0.1]], "m")

        store.add_chunks("a.md", ["second"], [[0.2]], "m", replace=False)

        assert [payload["text"] for payload in payloads(store)] == ["first", "second"]

    def test_positions_carry_on_across_batches(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["a", "b"], [[0.1], [0.2]], "m")
        store.add_chunks("a.md", ["c"], [[0.3]], "m", replace=False)
        store.add_chunks("a.md", ["d", "e"], [[0.4], [0.5]], "m", replace=False)

        assert [payload["position"] for payload in payloads(store)] == [0, 1, 2, 3, 4]

    def test_positions_stay_unique_per_source(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["a"], [[0.1]], "m")
        store.add_chunks("a.md", ["b"], [[0.2]], "m", replace=False)
        store.add_chunks("b.md", ["c"], [[0.3]], "m", replace=False)

        keys = [(payload["source"], payload["position"]) for payload in payloads(store)]
        assert len(keys) == len(set(keys))

    def test_each_source_has_its_own_counter(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["a1", "a2"], [[0.1], [0.2]], "m")

        store.add_chunks("b.md", ["b1"], [[0.3]], "m", replace=False)

        assert [(p["source"], p["position"]) for p in payloads(store)] == [("a.md", 0), ("a.md", 1), ("b.md", 0)]

    def test_first_batch_on_a_fresh_source_starts_at_zero(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["only"], [[0.1]], "m", replace=False)

        assert payloads(store)[0]["position"] == 0

    def test_replacing_after_appending_restarts_at_zero(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["a"], [[0.1]], "m")
        store.add_chunks("a.md", ["b"], [[0.2]], "m", replace=False)

        store.add_chunks("a.md", ["fresh"], [[0.3]], "m")

        assert [payload["position"] for payload in payloads(store)] == [0]


class TestAddChunksRejects:
    def test_more_chunks_than_vectors(self, store: QdrantStore) -> None:
        with pytest.raises(VectorStoreError, match="2 chunks for 1 vectors"):
            store.add_chunks("a.md", ["x", "y"], [[0.1]], "m")

    def test_more_vectors_than_chunks(self, store: QdrantStore) -> None:
        with pytest.raises(VectorStoreError):
            store.add_chunks("a.md", ["x"], [[0.1], [0.2]], "m")

    def test_vectors_of_mixed_dimensions(self, store: QdrantStore) -> None:
        with pytest.raises(VectorStoreError, match=r"inconsistent dimensions.*\[1, 2\]"):
            store.add_chunks("a.md", ["x", "y"], [[0.1], [0.1, 0.2]], "m")

    def test_a_dimension_the_collection_does_not_hold(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["x"], [[0.1, 0.2]], "m")

        with pytest.raises(VectorStoreError, match="holds 2-dimension vectors, not 3"):
            store.add_chunks("b.md", ["y"], [[0.1, 0.2, 0.3]], "m")

    def test_that_refusal_says_to_re_index(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["x"], [[0.1, 0.2]], "m")

        with pytest.raises(VectorStoreError, match="re-index every source"):
            store.add_chunks("b.md", ["y"], [[0.1, 0.2, 0.3]], "m")

    def test_writes_nothing_when_it_rejects(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["keep me"], [[0.1]], "m")

        with pytest.raises(VectorStoreError):
            store.add_chunks("a.md", ["x", "y"], [[0.1]], "m")

        assert [payload["text"] for payload in payloads(store)] == ["keep me"]

    def test_the_error_is_a_rag_error(self, store: QdrantStore) -> None:
        with pytest.raises(RagError):
            store.add_chunks("a.md", ["x", "y"], [[0.1]], "m")


class TestRemoveSource:
    def test_drops_the_chunks_of_that_source(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["x", "y"], [[0.1], [0.2]], "m")

        assert store.remove_source("a.md") == 2
        assert store.count() == 0

    def test_leaves_the_other_sources(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["x"], [[0.1]], "m")
        store.add_chunks("b.md", ["y"], [[0.2]], "m")

        store.remove_source("a.md")

        assert [payload["source"] for payload in payloads(store)] == ["b.md"]

    def test_an_unknown_source_drops_nothing(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["x"], [[0.1]], "m")

        assert store.remove_source("never-seen.md") == 0
        assert store.count() == 1

    def test_it_works_before_anything_was_written(self, store: QdrantStore) -> None:
        assert store.remove_source("a.md") == 0


class TestCountAndSources:
    def test_counts_everything_by_default(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["x", "y"], [[0.1], [0.2]], "m")
        store.add_chunks("b.md", ["z"], [[0.3]], "m")

        assert store.count() == 3

    def test_counts_one_source(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["x", "y"], [[0.1], [0.2]], "m")
        store.add_chunks("b.md", ["z"], [[0.3]], "m")

        assert store.count("a.md") == 2

    def test_lists_the_stored_sources(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["x"], [[0.1]], "m")
        store.add_chunks("b.md", ["y"], [[0.2]], "m")

        assert store.sources() == {"a.md", "b.md"}

    def test_lists_nothing_before_anything_was_written(self, store: QdrantStore) -> None:
        assert store.sources() == set()


class TestPayloadIndex:
    """The index on `source` only exists on a server: embedded Qdrant ignores it."""

    @staticmethod
    def _store_with_mock_client(location: str) -> tuple[QdrantStore, MagicMock]:
        """A store whose client is a mock reporting an empty database."""
        client = MagicMock()
        client.collection_exists.return_value = False
        client.scroll.return_value = ([], None)
        with patch("sensai.rag.qdrant_store.QdrantClient", return_value=client):
            return QdrantStore(location), client

    def test_a_server_gets_the_index_on_source(self) -> None:
        store, client = self._store_with_mock_client("http://localhost:6333")

        store.add_chunks("a.md", ["x"], [[0.1]], "m")

        client.create_payload_index.assert_called_once_with(
            DEFAULT_COLLECTION,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    def test_an_embedded_database_does_not(self, tmp_path: Path) -> None:
        store, client = self._store_with_mock_client(str(tmp_path / "qdrant"))

        store.add_chunks("a.md", ["x"], [[0.1]], "m")

        client.create_payload_index.assert_not_called()

    def test_an_in_memory_database_does_not(self) -> None:
        store, client = self._store_with_mock_client(":memory:")

        store.add_chunks("a.md", ["x"], [[0.1]], "m")

        client.create_payload_index.assert_not_called()

    def test_the_index_is_only_created_with_the_collection(self) -> None:
        store, client = self._store_with_mock_client("http://localhost:6333")
        store.add_chunks("a.md", ["x"], [[0.1]], "m")
        client.collection_exists.return_value = True
        client.get_collection.return_value.config.params.vectors.size = 1

        store.add_chunks("b.md", ["y"], [[0.2]], "m")

        assert client.create_payload_index.call_count == 1

    @pytest.mark.parametrize(
        ("location", "served"),
        [
            ("http://localhost:6333", True),
            ("https://qdrant.example:6333", True),
            (":memory:", False),
            ("rag_qdrant", False),
            ("/var/lib/sensai/qdrant", False),
        ],
    )
    def test_the_location_decides_the_mode(self, location: str, *, served: bool) -> None:
        store, _ = self._store_with_mock_client(location)

        assert store._served is served


class TestSearch:
    def test_the_closest_chunk_comes_first(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["far", "near"], [[0.0, 1.0], [1.0, 0.0]], "m")

        found = store.search([1.0, 0.05])

        assert [chunk["text"] for chunk in found] == ["near", "far"]

    def test_every_hit_carries_its_score(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["x"], [[1.0, 0.0]], "m")

        assert store.search([1.0, 0.0])[0]["score"] == pytest.approx(1.0)

    def test_every_hit_carries_the_stored_payload(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["hello"], [[1.0, 0.0]], "bge-m3")

        found = store.search([1.0, 0.0])[0]

        assert found["source"] == "a.md"
        assert found["position"] == 0
        assert found["text"] == "hello"
        assert found["model"] == "bge-m3"

    def test_the_limit_caps_the_hits(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["a", "b", "c"], [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], "m")

        assert len(store.search([1.0, 0.0], limit=2)) == 2

    def test_a_minimum_score_drops_the_far_ones(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["near", "far"], [[1.0, 0.0], [0.0, 1.0]], "m")

        found = store.search([1.0, 0.0], min_score=0.5)

        assert [chunk["text"] for chunk in found] == ["near"]

    def test_a_minimum_score_can_drop_everything(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["far"], [[0.0, 1.0]], "m")

        assert store.search([1.0, 0.0], min_score=0.9) == []

    def test_searching_an_empty_store_finds_nothing(self, store: QdrantStore) -> None:
        assert store.search([1.0, 0.0]) == []

    def test_a_vector_of_the_wrong_size_is_refused(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["x"], [[1.0, 0.0]], "m")

        with pytest.raises(VectorStoreError, match="3-dimension vector, but the collection holds 2"):
            store.search([1.0, 0.0, 0.0])

    def test_that_refusal_is_a_rag_error(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["x"], [[1.0, 0.0]], "m")

        with pytest.raises(RagError):
            store.search([1.0])

    def test_it_searches_across_every_source(self, store: QdrantStore) -> None:
        store.add_chunks("a.md", ["from a"], [[1.0, 0.0]], "m")
        store.add_chunks("b.md", ["from b"], [[0.9, 0.1]], "m")

        assert {chunk["source"] for chunk in store.search([1.0, 0.0])} == {"a.md", "b.md"}
