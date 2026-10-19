from sensai.rag.indexation_rag import Indexation


class FakeEmbedder:
    """An embedder whose vectors are deterministic and need no server.

    Texts mentioning "engine" land on one axis, the others on the opposite one,
    with a small drift per position so that consecutive scores are never exactly
    equal: an adaptive threshold sitting right on a score would find no
    breakpoint at all.
    """

    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.models: list[str] = []

    def embeddings(self, messages: list[str], model: str) -> list[list]:
        """Return one fake 2-D vector per message, and record the call."""
        self.batches.append(list(messages))
        self.models.append(model)
        vectors: list[list[float]] = []
        for i, message in enumerate(messages):
            drift = i * 0.01
            vectors.append([0.1 + drift, 1.0] if "engine" in message.lower() else [1.0, 0.1 + drift])
        return vectors


class SilentEmbedder:
    """An embedder that returns nothing, as Ollama does for an unavailable model."""

    def embeddings(self, messages: list[str], model: str) -> list[list]:
        """Return no vector at all."""
        del messages, model
        return []


class FakeIndexation(Indexation):
    """An Indexation wired to a `FakeEmbedder`, so no server is needed."""

    def __init__(self) -> None:
        """Build the indexation on a fake embedder."""
        super().__init__(FakeEmbedder())

    @property
    def batches(self) -> list[list[str]]:
        """The messages of every embedding call made so far."""
        return self._embedder.batches
