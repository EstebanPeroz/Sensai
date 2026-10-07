from pathlib import Path

import pytest

from sensai.memory.persistent_db.sql_alchemy.core import SqlAlchemy


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SqlAlchemy:
    """A SqlAlchemy bound to a fresh sqlite file (the class opens `test.db` in the cwd)."""
    monkeypatch.chdir(tmp_path)
    return SqlAlchemy()
