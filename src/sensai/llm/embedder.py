from __future__ import annotations

from typing import TYPE_CHECKING

from sensai.error import SensaiError

if TYPE_CHECKING:
    from sensai.llm.adapter import ProviderAdapter


class Embedder:
    """Embeds texts with one provider and model, shared by every feature comparing texts by meaning."""

    _provider: ProviderAdapter
    _model: str

    def __init__(self, provider: ProviderAdapter, model: str) -> None:
        """Bind the embedder to the `provider` serving the embedding `model`."""
        self._provider = provider
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text, or an empty list when the embedding model is unavailable or the call fails."""
        if not texts:
            return []
        try:
            vectors = self._provider.embeddings(texts, self._model)
        except SensaiError:
            return []
        if len(vectors) != len(texts):
            return []
        return vectors
