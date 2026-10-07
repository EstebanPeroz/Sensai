from typing import TYPE_CHECKING

from sensai.memory.persistent_db.error import DbConnectionError
from sensai.memory.persistent_db.sql_alchemy.core import SqlAlchemy

if TYPE_CHECKING:
    from sensai.memory.persistent_db.adapter import PersistentDatabase


class _State:
    db: PersistentDatabase | None = None


def set_db() -> None:
    """Set the database to the used db adapter."""
    if _State.db is not None:
        return
    try:
        _State.db = SqlAlchemy()
    except DbConnectionError:
        return


def get_db() -> PersistentDatabase | None:
    """Get the db."""
    if _State.db is None:
        set_db()
    return _State.db
