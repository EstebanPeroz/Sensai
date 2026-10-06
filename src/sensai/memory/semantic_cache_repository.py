from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CacheScope:
    """Context an answer was produced in: a cached answer is only served back within the same scope."""

    model: str
    persona: str
    subject: str | None = None


class SemanticCacheRepository(Protocol):
    """Storage of answers to past queries."""

    def lookup(self, query: str, scope: CacheScope, context: str = "") -> str | None:
        """Return the cached answer of the most similar query of the scope, asked after a similar context."""
        ...

    def store(self, query: str, scope: CacheScope, answer: str, context: str = "") -> None:
        """Cache the answer to a query asked after the given context, within its scope."""

    def clear(self) -> None:
        """Remove every cached answer."""
