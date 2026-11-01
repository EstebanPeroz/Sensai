from typing import ClassVar, override

import pytest
from sensai_server.app.app import _build_embedder
from sensai_server.config.settings import LLMSettings, OllamaSettings, Settings
from sensai_server.llm.embedder import Embedder
from sensai_server.llm.error import EmbeddingError, RequestCallError
from sensai_server.llm.provider import ProviderManager, provider_manager

from tests.server.fakes import FakeAdapter

EMBEDDING_MODEL = "embed"


class EmbeddingAdapter(FakeAdapter):
    """A `FakeAdapter` whose `embeddings` returns `vectors`, or raises when `failing_embeddings` is set."""

    vectors: ClassVar[list[list]] = []
    failing_embeddings: ClassVar[bool] = False

    def __init__(self, settings: OllamaSettings) -> None:
        super().__init__(settings)
        self.embedded: list[tuple[list[str], str]] = []

    @override
    def embeddings(self, messages: list[str], model: str) -> list[list]:
        self.embedded.append((messages, model))
        if self.failing_embeddings:
            msg = " Post -> connection refused"
            raise RequestCallError(msg)
        return self.vectors


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> EmbeddingAdapter:
    monkeypatch.setattr(EmbeddingAdapter, "vectors", [])
    monkeypatch.setattr(EmbeddingAdapter, "failing_embeddings", False)
    return EmbeddingAdapter(OllamaSettings())


class TestEmbedder:
    def test_model_is_the_bound_model(self, adapter: EmbeddingAdapter) -> None:
        assert Embedder(adapter, EMBEDDING_MODEL).model == EMBEDDING_MODEL

    def test_no_text_gives_no_vector_without_calling_the_provider(self, adapter: EmbeddingAdapter) -> None:
        assert Embedder(adapter, EMBEDDING_MODEL).embed([]) == []
        assert adapter.embedded == []

    def test_returns_one_vector_per_text(self, adapter: EmbeddingAdapter, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(EmbeddingAdapter, "vectors", [[1.0, 0.0], [0.0, 1.0]])

        assert Embedder(adapter, EMBEDDING_MODEL).embed(["x", "y"]) == [[1.0, 0.0], [0.0, 1.0]]
        assert adapter.embedded == [(["x", "y"], EMBEDDING_MODEL)]

    def test_no_vector_means_the_model_is_not_available(self, adapter: EmbeddingAdapter) -> None:
        with pytest.raises(EmbeddingError, match="'embed' is not available"):
            Embedder(adapter, EMBEDDING_MODEL).embed(["x"])

    def test_wrong_vector_count_is_an_error(self, adapter: EmbeddingAdapter, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(EmbeddingAdapter, "vectors", [[1.0]])

        with pytest.raises(EmbeddingError, match="returned 1 vectors for 2 texts"):
            Embedder(adapter, EMBEDDING_MODEL).embed(["x", "y"])

    def test_provider_error_becomes_an_embedding_error(
        self, adapter: EmbeddingAdapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(EmbeddingAdapter, "failing_embeddings", True)

        with pytest.raises(EmbeddingError, match="'embed' failed") as info:
            Embedder(adapter, EMBEDDING_MODEL).embed(["x"])
        assert isinstance(info.value.__cause__, RequestCallError)


class TestBuildEmbedder:
    @pytest.fixture(autouse=True)
    def serve_embedding_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(provider_manager.PROVIDER_ADAPTERS, "ollama", EmbeddingAdapter)
        monkeypatch.setattr(EmbeddingAdapter, "models", ["a", EMBEDDING_MODEL])

    @staticmethod
    def settings(embedding_model: str | None) -> Settings:
        return Settings(llm=LLMSettings(embedding_model=embedding_model, providers={"ollama": OllamaSettings()}))

    def test_no_embedding_model_gives_no_embedder(self) -> None:
        settings = self.settings(None)

        assert _build_embedder(settings, ProviderManager(settings.llm)) is None

    def test_unserved_embedding_model_gives_no_embedder(self) -> None:
        settings = self.settings("missing")

        assert _build_embedder(settings, ProviderManager(settings.llm)) is None

    def test_served_embedding_model_gives_an_embedder_on_its_provider(self) -> None:
        settings = self.settings(EMBEDDING_MODEL)
        pm = ProviderManager(settings.llm)

        embedder = _build_embedder(settings, pm)

        assert embedder is not None
        assert embedder.model == EMBEDDING_MODEL
        assert embedder._provider is pm.get_provider(EMBEDDING_MODEL)  # noqa: SLF001
