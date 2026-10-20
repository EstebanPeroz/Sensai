import pytest

from sensai.rag.error import EmbeddingError, VectorStoreError
from sensai.rag.indexation_rag import Indexation
from sensai.rag.qdrant_store import QdrantStore
from sensai.rag.retrievial_rag import Retrieval
from tests.rag.fakes import FailingEmbedder, FakeAnswerer, FakeIndexation, SilentAnswerer

ABOUT_ENGINES = "The engine burns fuel."
ABOUT_CATS = "The cat sleeps."


@pytest.fixture
def indexed(store: QdrantStore, indexation: FakeIndexation) -> Retrieval:
    """A Retrieval over a store holding one chunk on engines and one on cats."""
    store.add_chunks("engines.md", [ABOUT_ENGINES], indexation.vectorize_batch([ABOUT_ENGINES]), indexation.model)
    store.add_chunks("cats.md", [ABOUT_CATS], indexation.vectorize_batch([ABOUT_CATS]), indexation.model)
    return Retrieval(indexation, store, FakeAnswerer())


class TestResearch:
    def test_the_closest_chunk_comes_first(self, indexed: Retrieval) -> None:
        found = indexed.research("What about the engine?")

        assert found[0]["text"] == ABOUT_ENGINES

    def test_it_finds_the_other_topic_too(self, indexed: Retrieval) -> None:
        found = indexed.research("Tell me about the cat.")

        assert found[0]["text"] == ABOUT_CATS

    def test_every_hit_carries_its_score_and_source(self, indexed: Retrieval) -> None:
        found = indexed.research("What about the engine?")[0]

        assert found["source"] == "engines.md"
        # a cosine score can overshoot 1 by a float epsilon: Qdrant returned 1.0000000084
        assert found["score"] == pytest.approx(1.0, abs=1e-6)

    def test_the_limit_caps_the_hits(self, indexed: Retrieval) -> None:
        assert len(indexed.research("engine", limit=1)) == 1

    def test_a_minimum_score_can_drop_everything(self, indexed: Retrieval) -> None:
        assert indexed.research("engine", min_score=1.1) == []

    def test_the_question_is_embedded_once(self, indexed: Retrieval, indexation: FakeIndexation) -> None:
        indexation.batches.clear()

        indexed.research("What about the engine?")

        assert indexation.batches == [["What about the engine?"]]

    def test_it_asks_the_model_that_indexed_the_chunks(self, indexed: Retrieval, indexation: FakeIndexation) -> None:
        indexation.models.clear()

        indexed.research("engine")

        assert indexation.models == [indexation.model]

    def test_searching_an_empty_store_finds_nothing(self, store: QdrantStore, indexation: FakeIndexation) -> None:
        assert Retrieval(indexation, store, FakeAnswerer()).research("anything") == []


class TestResearchFailures:
    def test_an_embedding_failure_surfaces(self, store: QdrantStore) -> None:
        retrieval = Retrieval(Indexation(FailingEmbedder()), store, FakeAnswerer())

        with pytest.raises(EmbeddingError):
            retrieval.research("anything")

    def test_a_question_vector_of_the_wrong_size_is_refused(
        self, store: QdrantStore, indexation: FakeIndexation
    ) -> None:
        store.add_chunks("a.md", ["x"], [[0.1, 0.2, 0.3]], "m")  # 3 dimensions

        with pytest.raises(VectorStoreError, match="2-dimension vector, but the collection holds 3"):
            Retrieval(indexation, store, FakeAnswerer()).research("engine")  # the fake embeds in 2


class TestGetSources:
    def test_names_the_documents_of_the_hits(self, indexed: Retrieval) -> None:
        found = indexed.research("engine", min_score=None)  # no floor, so both topics come back

        assert set(indexed.get_sources(found)) == {"engines.md", "cats.md"}

    def test_the_best_match_comes_first(self, indexed: Retrieval) -> None:
        found = indexed.research("What about the engine?")

        assert indexed.get_sources(found)[0] == "engines.md"

    def test_a_source_is_named_once(self, indexed: Retrieval) -> None:
        chunks = [
            {"source": "a.md", "position": 0},
            {"source": "a.md", "position": 1},
            {"source": "b.md", "position": 0},
        ]

        assert indexed.get_sources(chunks) == ["a.md", "b.md"]

    def test_no_hit_names_no_source(self, indexed: Retrieval) -> None:
        assert indexed.get_sources([]) == []


class TestCreatePrompt:
    def test_it_carries_the_question(self, indexed: Retrieval) -> None:
        messages = indexed.create_prompt("Why?", [{"source": "a.md", "position": 0, "text": "Because."}])

        assert messages[-1] == {"role": "user", "content": "Why?"}

    def test_the_extracts_go_in_a_system_message(self, indexed: Retrieval) -> None:
        messages = indexed.create_prompt("Why?", [{"source": "a.md", "position": 0, "text": "Because."}])

        assert messages[0]["role"] == "system"
        assert "Because." in messages[0]["content"]

    def test_every_extract_is_numbered_and_sourced(self, indexed: Retrieval) -> None:
        chunks = [
            {"source": "a.md", "position": 3, "text": "First."},
            {"source": "b.md", "position": 0, "text": "Second."},
        ]

        system = indexed.create_prompt("Why?", chunks)[0]["content"]

        assert "[1] a.md (passage 3)" in system
        assert "[2] b.md (passage 0)" in system

    def test_the_model_is_told_not_to_guess(self, indexed: Retrieval) -> None:
        system = indexed.create_prompt("Why?", [{"source": "a.md", "position": 0, "text": "x"}])[0]["content"]

        assert "say so plainly instead of guessing" in system


class TestGeneration:
    def test_it_returns_what_the_model_said(self, store: QdrantStore, indexation: FakeIndexation) -> None:
        retrieval = Retrieval(indexation, store, FakeAnswerer("The rendement is low."))

        assert retrieval.generation([{"role": "user", "content": "x"}]) == "The rendement is low."

    def test_it_sends_the_current_model(self, store: QdrantStore, indexation: FakeIndexation) -> None:
        answerer = FakeAnswerer()
        retrieval = Retrieval(indexation, store, answerer)

        retrieval.generation([{"role": "user", "content": "x"}])

        assert answerer.payloads[0]["model"] == answerer.current_model()

    def test_it_sends_the_messages_it_was_given(self, store: QdrantStore, indexation: FakeIndexation) -> None:
        answerer = FakeAnswerer()
        messages = [{"role": "user", "content": "x"}]

        Retrieval(indexation, store, answerer).generation(messages)

        assert answerer.payloads[0]["messages"] == messages

    def test_an_error_from_the_model_is_reported(self, store: QdrantStore, indexation: FakeIndexation) -> None:
        retrieval = Retrieval(indexation, store, FakeAnswerer(error="model is gone"))

        assert "model is gone" in retrieval.generation([{"role": "user", "content": "x"}])

    def test_an_empty_answer_becomes_an_empty_string(self, store: QdrantStore, indexation: FakeIndexation) -> None:
        retrieval = Retrieval(indexation, store, SilentAnswerer())

        assert retrieval.generation([{"role": "user", "content": "x"}]) == ""


class TestAnswer:
    def test_it_answers_and_names_its_sources(self, indexed: Retrieval) -> None:
        answer, sources = indexed.answer("What about the engine?")

        assert answer == "A grounded answer."
        assert sources == ["engines.md"]

    def test_nothing_relevant_is_said_plainly(self, indexed: Retrieval) -> None:
        answer, sources = indexed.answer("engine", min_score=1.1)

        assert "nothing close enough" in answer
        assert sources == []

    def test_nothing_relevant_asks_the_model_nothing(self, store: QdrantStore, indexation: FakeIndexation) -> None:
        answerer = FakeAnswerer()

        Retrieval(indexation, store, answerer).answer("anything")

        assert answerer.payloads == []

    def test_the_chunks_reach_the_model(self, indexed: Retrieval, store: QdrantStore) -> None:
        del store
        indexed.answer("What about the engine?")

        system = indexed._answerer.payloads[0]["messages"][0]["content"]
        assert ABOUT_ENGINES in system
