from pathlib import Path

import pytest

from sensai.config.settings import DBSettings, LLMSettings, RagDBSettings, Settings, load_settings
from sensai.rag.indexation_rag import DEFAULT_EMBEDDING_MODEL, Indexation
from sensai.rag.store import VectorStore
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

    def test_the_shipped_model_is_not_the_english_only_one(self) -> None:
        # nomic-embed-text scores +0.03 between unrelated French topics: noise, not signal.
        assert load_settings().llm.embedding_model != "nomic-embed-text"


class TestIndexationUsesIt:
    @pytest.mark.parametrize("model", ["bge-m3", "some-other-model"])
    def test_the_chosen_model_reaches_the_embedder(self, model: str) -> None:
        embedder = FakeEmbedder()

        Indexation(embedder, chosen_model(Settings(llm=LLMSettings(embedding_model=model)))).vectorize_batch(["a"])

        assert embedder.models == [model]

    def test_the_stored_chunks_record_the_configured_model(self) -> None:
        indexation = Indexation(FakeEmbedder(), chosen_model(load_settings()))

        assert indexation.model == load_settings().llm.embedding_model


class TestRagDatabaseUrl:
    def test_the_store_writes_where_the_settings_say(self, tmp_path: Path) -> None:
        url = f"sqlite:///{tmp_path / 'chosen.db'}"

        VectorStore(Settings(db=DBSettings(rag=RagDBSettings(url=url))).db.rag.url).add_chunks(
            "a.md", ["x"], [[0.1]], "m"
        )

        assert (tmp_path / "chosen.db").exists()

    def test_the_shipped_settings_point_at_rag_db(self) -> None:
        assert load_settings().db.rag.url == "sqlite:///rag.db"

    def test_the_two_databases_are_not_the_same_file(self) -> None:
        settings = load_settings()

        assert settings.db.rag.url != settings.db.persistent.url
