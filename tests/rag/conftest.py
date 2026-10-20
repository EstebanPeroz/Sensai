from collections.abc import Iterator
from pathlib import Path

import pytest

from sensai.rag.qdrant_store import QdrantStore
from tests.rag.fakes import FakeIndexation


@pytest.fixture
def store_location(tmp_path: Path) -> str:
    """A fresh directory for an embedded Qdrant, to check what survives a new instance."""
    return str(tmp_path / "qdrant")


@pytest.fixture
def store(store_location: str) -> Iterator[QdrantStore]:
    """A QdrantStore on a fresh embedded database, closed once the test is done.

    Embedded Qdrant locks its directory for one client at a time, so a store
    left open would stop the next one in the same test from opening it.
    """
    opened = QdrantStore(store_location)
    yield opened
    opened.close()


@pytest.fixture
def indexation() -> FakeIndexation:
    """An Indexation with deterministic embeddings and no network call."""
    return FakeIndexation()
