from typing import TYPE_CHECKING

from sensai.memory.persistent_db.error import DbConnectionError
from sensai.memory.persistent_db.sql_alchemy.core import SqlAlchemy

if TYPE_CHECKING:
    from sensai.config.settings import PersistentDBSettings
    from sensai.memory.persistent_db.adapter import PersistentDatabase


class _State:
    db: PersistentDatabase | None = None


def set_db(settings: PersistentDBSettings) -> None:
    """Set the database to the used db adapter.

    Error:
        DbConnectionError: If the connection to the Db fail or no url is found in config

    """
    if _State.db is not None:
        return
    if settings.url is None:
        msg = "No url in settings"
        raise DbConnectionError(msg)
    _State.db = SqlAlchemy(settings.url)


def get_db() -> PersistentDatabase | None:
    """Get the db."""
    return _State.db
