from pathlib import Path

import pytest

from sensai.config.settings import DBSettings, LLMSettings, RagDBSettings, Settings, load_settings
from sensai.rag.collection import Collection, chunk_point
from sensai.rag.indexation_rag import DEFAULT_EMBEDDING_MODEL, Indexation
from tests.rag.fakes import FakeEmbedder


def chosen_model(settings: Settings) -> str:
    """The model `Core._build_indexer` would pick from `settings`."""
    return settings.llm.embedding_model or DEFAULT_EMBEDDING_MODEL


class TestChosenModel:
    def test_the_configured_model_wins(self) -> None:
        assert chosen_model(Settings(llm=LLMSettings(embedding_model="some-model"))) == "some-model"

    def test_without_one_the_default_is_used(self) -> None:
        assert chosen_model(Settings()) == DEFAULT_EMBEDDING_MODEL

    def test_the_shipped_settings_name_a_model(self) -> None:
        assert load_settings().llm.embedding_model is not None

    def test_the_shipped_model_is_a_model_ollama_can_serve(self) -> None:
        # The guard that used to forbid nomic-embed-text here was dropped on purpose:
        # it is now the configured model, chosen for its size over its French accuracy.
        assert load_settings().llm.embedding_model in {"nomic-embed-text", "bge-m3"}


class TestIndexationUsesIt:
    @pytest.mark.parametrize("model", ["nomic-embed-text", "some-other-model"])
    def test_the_chosen_model_reaches_the_embedder(self, model: str) -> None:
        embedder = FakeEmbedder()

        Indexation(embedder, chosen_model(Settings(llm=LLMSettings(embedding_model=model)))).vectorize_batch(["a"])

        assert embedder.models == [model]

    def test_the_stored_chunks_record_the_configured_model(self) -> None:
        indexation = Indexation(FakeEmbedder(), chosen_model(load_settings()))

        assert indexation.model == load_settings().llm.embedding_model


class TestRagDatabaseLocation:
    def test_the_collection_writes_where_the_settings_say(self, tmp_path: Path) -> None:
        location = str(tmp_path / "chosen")
        settings = Settings(db=DBSettings(rag=RagDBSettings(location=location)))

        collection = Collection.open(settings.db.rag.location)
        collection.upsert([chunk_point("a.md", 0, "x", [0.1], "m")])
        collection.close()

        assert (tmp_path / "chosen").exists()

    def test_the_shipped_settings_name_a_location(self) -> None:
        assert load_settings().db.rag.location == "http://localhost:6333"

    def test_the_two_databases_are_not_the_same_place(self) -> None:
        settings = load_settings()

        assert settings.db.rag.location != settings.db.persistent.url
