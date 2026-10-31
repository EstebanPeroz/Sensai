from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from sensai.memory.persistent_db.error import DatabaseError
from sensai.memory.persistent_db.sql_alchemy.table import SummaryTable
from sensai.memory.summary import Summary

if TYPE_CHECKING:
    from uuid import UUID


class SqlSummaryRepository:
    """`SummaryRepository` stored in the SQL database of the `SqlAlchemy` adapter, sharing its engine."""

    _engine: Engine

    def __init__(self, engine: Engine) -> None:
        """Store summaries through `engine`, whose tables already exist."""
        self._engine = engine

    def get(self, branch_id: UUID) -> Summary | None:
        """Return the summary of the branch `branch_id`, None when it has none.

        Raises:
            DatabaseError: If it could not be read.

        """
        with Session(self._engine) as session:
            try:
                row = session.scalars(select(SummaryTable).where(SummaryTable.branch_id == branch_id)).one_or_none()
            except SQLAlchemyError:
                msg = "the summary could not be read"
                raise DatabaseError(msg) from None
            return None if row is None else Summary(covered=row.covered, content=row.content)

    def save(self, branch_id: UUID, summary: Summary) -> None:
        """Make `summary` the summary of the branch `branch_id`, replacing the previous one.

        Raises:
            DatabaseError: If it could not be stored, such as when no branch has this id.

        """
        with Session(self._engine) as session:
            try:
                row = session.scalars(select(SummaryTable).where(SummaryTable.branch_id == branch_id)).one_or_none()
                if row is None:
                    row = SummaryTable(branch_id=branch_id)
                    session.add(row)
                row.covered = summary.covered
                row.content = summary.content
                session.commit()
            except SQLAlchemyError:
                msg = "the summary could not be stored"
                raise DatabaseError(msg) from None
