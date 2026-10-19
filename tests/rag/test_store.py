import pytest
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from sensai.rag.error import RagError, VectorStoreError
from sensai.rag.store import ChunkTable, VectorStore, dump_vector, load_vector


def rows(store: VectorStore) -> list[dict]:
    """Every stored chunk as a plain dict, ordered by source then position."""
    with Session(store._engine) as session:
        found = session.scalars(select(ChunkTable).order_by(ChunkTable.source, ChunkTable.position))
        return [
            {
                "source": chunk.source,
                "position": chunk.position,
                "text": chunk.text,
                "vector": chunk.vector,
                "model": chunk.model,
                "dim": chunk.dim,
            }
            for chunk in found
        ]


class TestInit:
    def test_creates_the_schema(self, store: VectorStore) -> None:
        assert inspect(store._engine).get_table_names() == ["chunk"]

    def test_indexes_source_and_model_only(self, store: VectorStore) -> None:
        indexed = {tuple(index["column_names"]) for index in inspect(store._engine).get_indexes("chunk")}
        assert indexed == {("source",), ("model",)}

    def test_starts_empty(self, store: VectorStore) -> None:
        assert rows(store) == []

    def test_data_survives_a_new_instance(self, store_url: str) -> None:
        VectorStore(store_url).add_chunks("a.md", ["x"], [[0.1]], "m")

        assert [row["text"] for row in rows(VectorStore(store_url))] == ["x"]


class TestVectorSerialisation:
    def test_round_trip_keeps_the_values(self) -> None:
        vector = [0.1, -0.25, 0.0, 1.5]

        assert load_vector(dump_vector(vector)) == vector

    def test_dumps_to_readable_json(self) -> None:
        assert dump_vector([0.1, 0.2]) == "[0.1, 0.2]"


class TestAddChunks:
    def test_returns_how_many_were_written(self, store: VectorStore) -> None:
        assert store.add_chunks("a.md", ["x", "y"], [[0.1], [0.2]], "m") == 2

    def test_stores_text_vector_model_and_dimension(self, store: VectorStore) -> None:
        store.add_chunks("a.md", ["hello"], [[0.1, 0.2, 0.3]], "bge-m3")

        assert rows(store) == [
            {
                "source": "a.md",
                "position": 0,
                "text": "hello",
                "vector": "[0.1, 0.2, 0.3]",
                "model": "bge-m3",
                "dim": 3,
            }
        ]

    def test_positions_follow_the_list_order(self, store: VectorStore) -> None:
        store.add_chunks("a.md", ["first", "second", "third"], [[0.1], [0.2], [0.3]], "m")

        assert [(row["position"], row["text"]) for row in rows(store)] == [
            (0, "first"),
            (1, "second"),
            (2, "third"),
        ]

    def test_replaces_the_previous_chunks_of_the_source(self, store: VectorStore) -> None:
        store.add_chunks("a.md", ["old", "stale"], [[0.1], [0.2]], "m")

        store.add_chunks("a.md", ["fresh"], [[0.3]], "m")

        assert [row["text"] for row in rows(store)] == ["fresh"]

    def test_leaves_the_other_sources_alone(self, store: VectorStore) -> None:
        store.add_chunks("a.md", ["from a"], [[0.1]], "m")
        store.add_chunks("b.md", ["from b"], [[0.2]], "m")

        store.add_chunks("a.md", ["new a"], [[0.3]], "m")

        assert [(row["source"], row["text"]) for row in rows(store)] == [("a.md", "new a"), ("b.md", "from b")]

    def test_empty_list_writes_nothing(self, store: VectorStore) -> None:
        assert store.add_chunks("a.md", [], [], "m") == 0
        assert rows(store) == []

    def test_empty_list_does_not_delete_what_is_stored(self, store: VectorStore) -> None:
        store.add_chunks("a.md", ["keep me"], [[0.1]], "m")

        store.add_chunks("a.md", [], [], "m")

        assert [row["text"] for row in rows(store)] == ["keep me"]


class TestAddChunksWithoutReplace:
    def test_appends_instead_of_replacing(self, store: VectorStore) -> None:
        store.add_chunks("a.md", ["first"], [[0.1]], "m")

        store.add_chunks("a.md", ["second"], [[0.2]], "m", replace=False)

        assert [row["text"] for row in rows(store)] == ["first", "second"]

    def test_positions_carry_on_across_batches(self, store: VectorStore) -> None:
        store.add_chunks("a.md", ["a", "b"], [[0.1], [0.2]], "m")
        store.add_chunks("a.md", ["c"], [[0.3]], "m", replace=False)
        store.add_chunks("a.md", ["d", "e"], [[0.4], [0.5]], "m", replace=False)

        assert [row["position"] for row in rows(store)] == [0, 1, 2, 3, 4]

    def test_positions_stay_unique_per_source(self, store: VectorStore) -> None:
        store.add_chunks("a.md", ["a"], [[0.1]], "m")
        store.add_chunks("a.md", ["b"], [[0.2]], "m", replace=False)
        store.add_chunks("b.md", ["c"], [[0.3]], "m", replace=False)

        keys = [(row["source"], row["position"]) for row in rows(store)]
        assert len(keys) == len(set(keys))

    def test_each_source_has_its_own_counter(self, store: VectorStore) -> None:
        store.add_chunks("a.md", ["a1", "a2"], [[0.1], [0.2]], "m")

        store.add_chunks("b.md", ["b1"], [[0.3]], "m", replace=False)

        assert [(row["source"], row["position"]) for row in rows(store)] == [
            ("a.md", 0),
            ("a.md", 1),
            ("b.md", 0),
        ]

    def test_first_batch_on_a_fresh_source_starts_at_zero(self, store: VectorStore) -> None:
        store.add_chunks("a.md", ["only"], [[0.1]], "m", replace=False)

        assert rows(store)[0]["position"] == 0

    def test_replacing_after_appending_restarts_at_zero(self, store: VectorStore) -> None:
        store.add_chunks("a.md", ["a"], [[0.1]], "m")
        store.add_chunks("a.md", ["b"], [[0.2]], "m", replace=False)

        store.add_chunks("a.md", ["fresh"], [[0.3]], "m")

        assert [row["position"] for row in rows(store)] == [0]


class TestAddChunksRejects:
    def test_more_chunks_than_vectors(self, store: VectorStore) -> None:
        with pytest.raises(VectorStoreError, match="2 chunks for 1 vectors"):
            store.add_chunks("a.md", ["x", "y"], [[0.1]], "m")

    def test_more_vectors_than_chunks(self, store: VectorStore) -> None:
        with pytest.raises(VectorStoreError):
            store.add_chunks("a.md", ["x"], [[0.1], [0.2]], "m")

    def test_vectors_of_mixed_dimensions(self, store: VectorStore) -> None:
        with pytest.raises(VectorStoreError, match=r"inconsistent dimensions.*\[1, 2\]"):
            store.add_chunks("a.md", ["x", "y"], [[0.1], [0.1, 0.2]], "m")

    def test_writes_nothing_when_it_rejects(self, store: VectorStore) -> None:
        store.add_chunks("a.md", ["keep me"], [[0.1]], "m")

        with pytest.raises(VectorStoreError):
            store.add_chunks("a.md", ["x", "y"], [[0.1]], "m")

        assert [row["text"] for row in rows(store)] == ["keep me"]

    def test_the_error_is_a_rag_error(self, store: VectorStore) -> None:
        with pytest.raises(RagError):
            store.add_chunks("a.md", ["x", "y"], [[0.1]], "m")
