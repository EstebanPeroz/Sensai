from pathlib import Path

import pytest

from sensai.rag.store import VectorStore
from tests.rag.fakes import FakeIndexation


@pytest.fixture
def store_url(tmp_path: Path) -> str:
    """The URL of a fresh sqlite file, to check what survives a new instance."""
    return f"sqlite:///{tmp_path / 'rag.db'}"


@pytest.fixture
def store(store_url: str) -> VectorStore:
    """A VectorStore on a fresh sqlite file, as the application uses one.

    Not "sqlite:///:memory:": every connection to it is its own empty database,
    so a store written from one thread looks untouched from another, which is
    not how the application runs.
    """
    return VectorStore(store_url)


@pytest.fixture
def indexation() -> FakeIndexation:
    """An Indexation with deterministic embeddings and no network call."""
    return FakeIndexation()
