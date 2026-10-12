from pathlib import Path

import pytest

from sensai.memory.persistent_db.sql_alchemy.core import SqlAlchemy


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    """The URL of a fresh sqlite file."""
    return f"sqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def db(db_url: str) -> SqlAlchemy:
    """A SqlAlchemy bound to a fresh sqlite file."""
    return SqlAlchemy(db_url)
