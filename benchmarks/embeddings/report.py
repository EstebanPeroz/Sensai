"""Tables of the results of the embedding model benchmark, one column or row per model."""

from __future__ import annotations

from typing import Any

from benchmarks.markdown import MISSING, table


def _value(value: float | None, unit: str = "") -> str:
    return MISSING if value is None else f"{value:g}{unit}"


def _share(results: list[dict[str, Any]], key: str) -> str:
    if not results:
        return MISSING
    return f"{sum(bool(result[key]) for result in results)}/{len(results)}"


def _mrr(results: list[dict[str, Any]]) -> str:
    """Mean reciprocal rank of the right passages: 1 when each one comes first, 0.5 when second..."""
    if not results:
        return MISSING
    return f"{sum(1 / result['rank'] for result in results) / len(results):.2f}"


def overview(results: list[dict[str, Any]]) -> list[str]:
    """One row per model: its size and speed, how well it finds passages and how it separates cached questions."""
    rows = []
    for result in results:
        cost = result.get("cost", {})
        retrieval = result.get("retrieval", [])
        cross = [query for query in retrieval if query["cross_lingual"]]
        cache = result.get("separation", {})
        rows.append(
            [
                f"`{result['model']}`",
                _value(result.get("size_gb"), " GB"),
                _value(cost.get("dimension")),
                _value(cost.get("cold_s"), " s"),
                _value(cost.get("passages_per_s"), "/s"),
                _share(retrieval, "top"),
                _share(retrieval, "kept"),
                _mrr(retrieval),
                _share(cross, "top"),
                _value(cache.get("margin")),
                _right(cache),
            ]
        )
    header = [
        "Model",
        "Size",
        "Dimension",
        "Cold load",
        "Passages",
        "Right passage first",
        "Right passage kept",
        "MRR",
        "Across languages",
        "Cache margin",
        "Cache pairs right (best threshold)",
    ]
    return table(header, rows)


def _right(cache: dict[str, Any]) -> str:
    if not cache:
        return MISSING
    return f"{cache['best_right']}/{cache['pairs']} ({cache['best_threshold']:g})"


def cache(results: list[dict[str, Any]]) -> list[str]:
    """One row per model: the scores of rephrased and different questions, and the cache at its threshold."""
    rows = []
    for result in results:
        stats = result.get("separation", {})
        rows.append(
            [
                f"`{result['model']}`",
                _value(stats.get("same_mean")),
                _value(stats.get("different_mean")),
                _value(stats.get("margin")),
                f"{stats['hits']}/{stats['pairs'] // 2}" if stats else MISSING,
                f"{stats['false_hits']}/{stats['pairs'] // 2}" if stats else MISSING,
                _right(stats),
            ]
        )
    threshold = next((result["separation"]["threshold"] for result in results if result.get("separation")), None)
    header = [
        "Model",
        "Rephrased (mean)",
        "Different (mean)",
        "Margin",
        f"Hits at {_value(threshold)}",
        f"False hits at {_value(threshold)}",
        "Right at best threshold",
    ]
    return table(header, rows)


def ranks(results: list[dict[str, Any]]) -> list[str]:
    """One row per query, one column per model, with the rank of its right passage among every passage."""
    columns = [{query["id"]: query for query in result.get("retrieval", [])} for result in results]
    ids = list(dict.fromkeys(query_id for column in columns for query_id in column))
    rows = []
    for query_id in ids:
        cross = next(column[query_id]["cross_lingual"] for column in columns if query_id in column)
        row = [query_id + (" (across languages)" if cross else "")]
        row += [str(column[query_id]["rank"]) if query_id in column else MISSING for column in columns]
        rows.append(row)
    return table(["Query", *(f"`{result['model']}`" for result in results)], rows)


def pairs(results: list[dict[str, Any]]) -> list[str]:
    """One row per pair of questions, one column per model, with their cosine similarity."""
    columns = [{pair["id"]: pair for pair in result.get("cache", [])} for result in results]
    ids = list(dict.fromkeys(pair_id for column in columns for pair_id in column))
    rows = []
    for pair_id in ids:
        pair = next(column[pair_id] for column in columns if pair_id in column)
        row = [f"{pair['first']} / {pair['second']}", "same" if pair["same"] else "different"]
        row += [f"{column[pair_id]['score']:.3f}" if pair_id in column else MISSING for column in columns]
        rows.append(row)
    return table(["Pair", "Meaning", *(f"`{result['model']}`" for result in results)], rows)


def render(results: list[dict[str, Any]]) -> str:
    """Render every table of the benchmark for `results`, one per model."""
    sections = [
        ("Overview", overview(results)),
        ("Semantic cache", cache(results)),
        ("Rank of the right passage", ranks(results)),
        ("Similarity of the pairs of questions", pairs(results)),
    ]
    lines = ["# Embedding model benchmark results", "", "Generated by `make bench-embeddings-report`."]
    for title, lines_of_table in sections:
        lines += ["", f"## {title}", "", *lines_of_table]
    return "\n".join(lines) + "\n"
