from pathlib import Path

import pytest

from sensai.rag.store import VectorStore
from tests.rag.fakes import FakeIndexation


@pytest.fixture
def store() -> VectorStore:
    """A VectorStore backed by an in-memory database."""
    return VectorStore("sqlite:///:memory:")


@pytest.fixture
def store_url(tmp_path: Path) -> str:
    """The URL of a fresh sqlite file, to check what survives a new instance."""
    return f"sqlite:///{tmp_path / 'rag.db'}"


@pytest.fixture
def indexation() -> FakeIndexation:
    """An Indexation with deterministic embeddings and no network call."""
    return FakeIndexation()
