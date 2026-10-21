from collections.abc import Iterator
from pathlib import Path

import pytest

from sensai.rag.collection import Collection
from tests.rag.fakes import FakeIndexation


@pytest.fixture
def store_location(tmp_path: Path) -> str:
    """A fresh directory for an embedded Qdrant, to check what survives a new client."""
    return str(tmp_path / "qdrant")


@pytest.fixture
def collection(store_location: str) -> Iterator[Collection]:
    """A Collection on a fresh embedded database, closed once the test is done.

    Embedded Qdrant locks its directory for one client at a time, so a
    collection left open would stop the next one in the same test from opening
    the same location.
    """
    opened = Collection.open(store_location)
    yield opened
    opened.close()


@pytest.fixture
def indexation() -> FakeIndexation:
    """An Indexation with deterministic embeddings and no network call."""
    return FakeIndexation()
