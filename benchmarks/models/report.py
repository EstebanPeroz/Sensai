"""Tables of the results of the model benchmark, one column or row per model."""

from __future__ import annotations

import statistics
from typing import Any

from benchmarks.markdown import MISSING, table

PASSED, FAILED = "✓", "✗"


def share(results: list[dict[str, Any]], key: str = "passed") -> str:
    """Tell how many of `results` have `key` true, as `passed/total`."""
    if not results:
        return MISSING
    return f"{sum(bool(result.get(key)) for result in results)}/{len(results)}"


def mean_seconds(results: list[dict[str, Any]]) -> str:
    """Mean time of the answers of `results`, in seconds."""
    if not results:
        return MISSING
    return f"{statistics.mean(result['seconds'] for result in results):.1f} s"


def _value(value: float | None, unit: str = "") -> str:
    return MISSING if value is None else f"{value:g}{unit}"


def _percent(value: float | None) -> str:
    return MISSING if value is None else f"{value:.0%}"


def overview(results: list[dict[str, Any]]) -> list[str]:
    """One row per model: its size, its speed and its score in each category."""
    rows = []
    for result in results:
        latency = result.get("latency", {})
        tools = result.get("tools", [])
        memory = result.get("memory", {})
        rows.append(
            [
                f"`{result['model']}`",
                _value(result.get("size_gb"), " GB"),
                _percent(result.get("gpu_share")),
                _value(latency.get("short", {}).get("completion_tokens_per_s"), " tok/s"),
                share(result.get("quality", [])),
                share(tools, "right_tool"),
                share(tools),
                share(result.get("instructions", [])),
                share(memory.get("full", [])),
                share(memory.get("compressed", [])),
                mean_seconds(result.get("quality", []) + result.get("instructions", [])),
            ]
        )
    header = [
        "Model",
        "Size",
        "On GPU",
        "Generation",
        "Answers",
        "Right tool",
        "Tool cases passed",
        "Instructions",
        "Memory (full)",
        "Memory (compressed)",
        "Mean answer time",
    ]
    return table(header, rows)


def latency(results: list[dict[str, Any]]) -> list[str]:
    """One row per model: cold load, then the warm medians alone and after a long context."""
    rows = []
    for result in results:
        measures = result.get("latency", {})
        cold, short, long = (measures.get(key, {}) for key in ("cold", "short", "long"))
        rows.append(
            [
                f"`{result['model']}`",
                _value(cold.get("load_s"), " s"),
                _value(cold.get("first_token_s"), " s"),
                _value(short.get("first_token_s"), " s"),
                _value(short.get("completion_tokens_per_s"), " tok/s"),
                _value(long.get("prompt_tokens"), " tok"),
                _value(long.get("prompt_tokens_per_s"), " tok/s"),
                _value(long.get("first_token_s"), " s"),
            ]
        )
    header = [
        "Model",
        "Cold load",
        "Cold first token",
        "Warm first token",
        "Generation",
        "Long context",
        "Prompt reading",
        "First token after it",
    ]
    return table(header, rows)


def per_case(results: list[dict[str, Any]], category: str) -> list[str]:
    """One row per case of `category`, one column per model, marking whether it passed."""
    outcomes = [{case["id"]: case["passed"] for case in result.get(category, [])} for result in results]
    ids = list(dict.fromkeys(case_id for outcome in outcomes for case_id in outcome))
    rows = [[case_id] + [_mark(outcome.get(case_id)) for outcome in outcomes] for case_id in ids]
    return table(["Case", *(f"`{result['model']}`" for result in results)], rows)


def tool_detail(results: list[dict[str, Any]]) -> list[str]:
    """One row per tool case, one column per model, with the tools it called."""
    columns = [{case["id"]: case for case in result.get("tools", [])} for result in results]
    ids = list(dict.fromkeys(case_id for column in columns for case_id in column))
    rows = []
    for case_id in ids:
        expected = next((column[case_id]["expected_tool"] for column in columns if case_id in column), None)
        row = [case_id, expected or "none"]
        for column in columns:
            case = column.get(case_id)
            if case is None:
                row.append(MISSING)
                continue
            called = ", ".join(case["called"]) or "none"
            row.append(f"{_mark(case['passed'])} {called}")
        rows.append(row)
    return table(["Case", "Expected tool", *(f"`{result['model']}`" for result in results)], rows)


def compression(results: list[dict[str, Any]]) -> list[str]:
    """One row per model: what the compression summarised, and the context the recall questions then took."""
    rows = []
    for result in results:
        memory = result.get("memory", {})
        stats = memory.get("compression", {})
        full = [case["prompt_tokens"] for case in memory.get("full", []) if case.get("prompt_tokens")]
        compressed = [case["prompt_tokens"] for case in memory.get("compressed", []) if case.get("prompt_tokens")]
        rows.append(
            [
                f"`{result['model']}`",
                _value(stats.get("messages")),
                _value(stats.get("tokens_before"), " tok"),
                _value(stats.get("tokens_after"), " tok"),
                _value(stats.get("seconds"), " s"),
                _value(statistics.median(full) if full else None, " tok"),
                _value(statistics.median(compressed) if compressed else None, " tok"),
            ]
        )
    header = [
        "Model",
        "Messages summarised",
        "Summarised text",
        "Summary",
        "Compression time",
        "Context (full)",
        "Context (compressed)",
    ]
    return table(header, rows)


def render(results: list[dict[str, Any]]) -> str:
    """Render every table of the benchmark for `results`, one per model."""
    sections = [
        ("Overview", overview(results)),
        ("Latency", latency(results)),
        ("Answers", per_case(results, "quality")),
        ("Tool use", tool_detail(results)),
        ("Instructions", per_case(results, "instructions")),
        ("Compression", compression(results)),
    ]
    lines = ["# Model benchmark results", "", "Generated by `make bench-report` from the JSON files of this folder."]
    for title, lines_of_table in sections:
        lines += ["", f"## {title}", "", *lines_of_table]
    return "\n".join(lines) + "\n"


def _mark(passed: bool | None) -> str:  # noqa: FBT001 - a value of the table, not an option
    if passed is None:
        return MISSING
    return PASSED if passed else FAILED
