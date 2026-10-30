"""Runs of the embedding model benchmark, each one through Sensai's own embedder and cosine similarity."""

from __future__ import annotations

import statistics
import time
from typing import TYPE_CHECKING, Any

from sensai.tools.web._shaping import cosine
from sensai.tools.web.fetch import KEPT_CHUNKS

if TYPE_CHECKING:
    from collections.abc import Callable

    from sensai.llm.adapter import ProviderAdapter
    from sensai.llm.embedder import Embedder


class EmbeddingBenchmark:
    """Score the embedding model of `embedder` on the cases of the benchmark."""

    def __init__(self, embedder: Embedder, cases: dict[str, Any], *, progress: Callable[[str], object]) -> None:
        """Embed the texts of `cases` with `embedder`, telling `progress` what it does."""
        self._embedder = embedder
        self._cases = cases
        self._progress = progress

    def cost(self, provider: ProviderAdapter) -> dict[str, Any]:
        """Time the first request after unloading the model, then the median of batches as the RAG indexation sends.

        Raises:
            EmbeddingError: If the model cannot embed.

        """
        case = self._cases["cost"]
        texts = [passage["text"] for passage in self._cases["passages"]]
        batches = [texts[i : i + case["batch"]] for i in range(0, len(texts), case["batch"])]
        provider.unload(self._embedder.model)
        start = time.perf_counter()
        dimension = len(self._embedder.embed([texts[0]])[0])
        cold = time.perf_counter() - start
        runs = []
        for _ in range(case["runs"]):
            start = time.perf_counter()
            for batch in batches:
                self._embedder.embed(batch)
            runs.append(time.perf_counter() - start)
        seconds = statistics.median(runs)
        self._progress(f"[cost] cold: {cold:.2f} s, {len(texts) / seconds:.1f} passages/s")
        return {
            "dimension": dimension,
            "cold_s": round(cold, 2),
            "passages": len(texts),
            "batch_s": round(seconds / len(batches), 3),
            "passages_per_s": round(len(texts) / seconds, 1),
        }

    def retrieval(self) -> list[dict[str, Any]]:
        """Rank every passage for each query, as `web_fetch` ranks the chunks of a page, and find the right one.

        Raises:
            EmbeddingError: If the model cannot embed.

        """
        passages = self._cases["passages"]
        queries = self._cases["queries"]
        passage_vectors = self._embedder.embed([passage["text"] for passage in passages])
        query_vectors = self._embedder.embed([query["prompt"] for query in queries])
        index = {passage["id"]: i for i, passage in enumerate(passages)}
        results = []
        for query, query_vector in zip(queries, query_vectors, strict=True):
            scores = [cosine(query_vector, vector) for vector in passage_vectors]
            ranking = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
            right = index[query["passage"]]
            rank = ranking.index(right) + 1
            result = {
                "id": query["id"],
                "rank": rank,
                "top": rank == 1,
                "kept": rank <= KEPT_CHUNKS,
                "cross_lingual": query["language"] != passages[right]["language"],
                "best": passages[ranking[0]]["id"],
                "right_score": _round(scores[right]),
                "best_score": _round(scores[ranking[0]]),
            }
            self._progress(f"[retrieval] {'ok  ' if rank == 1 else 'MISS'} {query['id']} (rank {rank})")
            results.append(result)
        return results

    def cache(self) -> list[dict[str, Any]]:
        """Score each pair of questions as the semantic cache compares a new query with a cached one.

        Raises:
            EmbeddingError: If the model cannot embed.

        """
        results = []
        for number, pair in enumerate(self._cases["pairs"], start=1):
            first, second = self._embedder.embed([pair["first"], pair["second"]])
            score = cosine(first, second)
            results.append(
                {
                    "id": f"p{number}",
                    "first": pair["first"],
                    "second": pair["second"],
                    "same": pair["same"],
                    "score": _round(score),
                }
            )
        self._progress(f"[cache] {len(results)} pairs scored")
        return results


def separation(pairs: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """How well the scores of `pairs` tell rephrased questions from different ones, at `threshold` and at best.

    The best threshold is the one sorting the most pairs right; it is only known after the fact, from these pairs.
    """
    same = [pair["score"] for pair in pairs if pair["same"]]
    different = [pair["score"] for pair in pairs if not pair["same"]]
    candidates = sorted({pair["score"] for pair in pairs})
    best = max(candidates, key=lambda candidate: (_right(pairs, candidate), -candidate)) if candidates else None
    return {
        "same_mean": _round(statistics.mean(same)) if same else None,
        "different_mean": _round(statistics.mean(different)) if different else None,
        "margin": _round(min(same) - max(different)) if same and different else None,
        "threshold": threshold,
        "hits": sum(score >= threshold for score in same),
        "false_hits": sum(score >= threshold for score in different),
        "best_threshold": best,
        "best_right": _right(pairs, best) if best is not None else None,
        "pairs": len(pairs),
    }


def _right(pairs: list[dict[str, Any]], threshold: float) -> int:
    """Count the pairs sorted right at `threshold`: rephrased ones at or above it, different ones below it."""
    return sum((pair["score"] >= threshold) == pair["same"] for pair in pairs)


def _round(value: float) -> float:
    return round(value, 4)
