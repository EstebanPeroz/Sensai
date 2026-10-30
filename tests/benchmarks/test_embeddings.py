from benchmarks.embeddings.report import render
from benchmarks.embeddings.runner import EmbeddingBenchmark, separation

VECTORS = {
    "compost text": [1.0, 0.0, 0.0],
    "bread text": [0.0, 1.0, 0.0],
    "wifi text": [0.0, 0.0, 1.0],
    "what to compost?": [0.9, 0.1, 0.0],
    "comment cuire le pain ?": [0.0, 0.2, 0.9],
    "a": [1.0, 0.0, 0.0],
    "a again": [0.99, 0.1, 0.0],
    "b": [0.0, 1.0, 0.0],
}
CASES = {
    "cost": {"batch": 2, "runs": 1},
    "passages": [
        {"id": "compost", "language": "en", "text": "compost text"},
        {"id": "bread", "language": "en", "text": "bread text"},
        {"id": "wifi", "language": "en", "text": "wifi text"},
    ],
    "queries": [
        {"id": "compost-q", "prompt": "what to compost?", "passage": "compost", "language": "en"},
        {"id": "bread-fr", "prompt": "comment cuire le pain ?", "passage": "bread", "language": "fr"},
    ],
    "pairs": [{"first": "a", "second": "a again", "same": True}, {"first": "a", "second": "b", "same": False}],
}


class FakeEmbedder:
    model = "fake-embed"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [VECTORS[text] for text in texts]


class FakeProvider:
    def __init__(self) -> None:
        self.unloaded: list[str] = []

    def unload(self, model: str) -> bool:
        self.unloaded.append(model)
        return True


def bench(embedder: FakeEmbedder | None = None) -> EmbeddingBenchmark:
    return EmbeddingBenchmark(embedder or FakeEmbedder(), CASES, progress=lambda _line: None)  # type: ignore[arg-type]


def test_retrieval_ranks_the_right_passage_among_every_passage() -> None:
    results = {result["id"]: result for result in bench().retrieval()}

    assert results["compost-q"]["rank"] == 1
    assert results["compost-q"]["top"]
    assert results["bread-fr"]["rank"] == 2
    assert results["bread-fr"]["best"] == "wifi"
    assert results["bread-fr"]["kept"]


def test_retrieval_tells_a_query_asked_in_another_language_than_its_passage() -> None:
    results = {result["id"]: result for result in bench().retrieval()}

    assert results["bread-fr"]["cross_lingual"]
    assert not results["compost-q"]["cross_lingual"]


def test_cost_unloads_the_model_then_sends_the_passages_in_batches() -> None:
    embedder, provider = FakeEmbedder(), FakeProvider()

    cost = bench(embedder).cost(provider)  # type: ignore[arg-type]

    assert provider.unloaded == ["fake-embed"]
    assert cost["dimension"] == 3
    assert embedder.calls[1:] == [["compost text", "bread text"], ["wifi text"]]


def test_separation_counts_the_hits_and_false_hits_at_the_threshold() -> None:
    pairs = [
        {"same": True, "score": 0.97},
        {"same": True, "score": 0.9},
        {"same": False, "score": 0.96},
        {"same": False, "score": 0.5},
    ]

    stats = separation(pairs, 0.95)

    assert stats["hits"] == 1
    assert stats["false_hits"] == 1
    assert stats["margin"] == -0.06
    assert stats["best_right"] == 3


def test_render_holds_every_table() -> None:
    benchmark = bench()
    pairs = benchmark.cache()
    result = {
        "model": "fake-embed",
        "size_gb": 0.1,
        "cost": {"dimension": 3, "cold_s": 1.0, "passages_per_s": 50.0},
        "retrieval": benchmark.retrieval(),
        "cache": pairs,
        "separation": separation(pairs, 0.95),
    }

    text = render([result])

    assert "| `fake-embed` | 0.1 GB | 3 | 1 s | 50/s | 1/2 | 2/2 | 0.75 | 0/1 |" in text
    assert "bread-fr (across languages) | 2 |" in text
    assert "a / b | different |" in text
