from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sensai.memory.persistent_db.adapter import PersistentDatabase

db: PersistentDatabase | None = None


def set_db() -> None:
    """Set the Databse to the used db adapter."""
    if db is None:
        return
    ### init


def get_db() -> PersistentDatabase | None:
    """Get the db."""
    if db is None:
        set_db()
    return db
