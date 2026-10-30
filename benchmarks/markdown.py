"""Markdown tables of the benchmark reports."""

from __future__ import annotations

MISSING = "-"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    """Lines of a markdown table with `header` and `rows`."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines
