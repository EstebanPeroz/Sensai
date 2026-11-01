from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from sensai_server.memory.persistent_db.error import DatabaseError
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai_server.memory.summary import Summary
from sqlalchemy.exc import SQLAlchemyError

from tests.server.conftest import MODEL, db_url


@pytest.fixture
def db(tmp_path: Path) -> SqlAlchemy:
    return SqlAlchemy(db_url(tmp_path))


def _branch(db: SqlAlchemy) -> UUID:
    branch = db.create_branch(model=MODEL)
    assert branch is not None
    return branch


class TestSummaries:
    def test_a_branch_starts_without_summary(self, db: SqlAlchemy) -> None:
        assert db.summaries.get(_branch(db)) is None

    def test_round_trip(self, db: SqlAlchemy) -> None:
        branch = _branch(db)

        db.summaries.save(branch, Summary(covered=4, content="The user is Ada, she wants a budget of 300 EUR."))

        assert db.summaries.get(branch) == Summary(covered=4, content="The user is Ada, she wants a budget of 300 EUR.")

    def test_a_summary_survives_a_reopen(self, tmp_path: Path, db: SqlAlchemy) -> None:
        branch = _branch(db)

        db.summaries.save(branch, Summary(covered=4, content="first"))

        assert SqlAlchemy(db_url(tmp_path)).summaries.get(branch) == Summary(covered=4, content="first")

    def test_a_new_summary_replaces_the_previous_one(self, db: SqlAlchemy) -> None:
        branch = _branch(db)

        db.summaries.save(branch, Summary(covered=4, content="first"))
        db.summaries.save(branch, Summary(covered=10, content="second"))

        assert db.summaries.get(branch) == Summary(covered=10, content="second")

    def test_each_branch_has_its_own(self, db: SqlAlchemy) -> None:
        first, second = _branch(db), _branch(db)

        db.summaries.save(first, Summary(covered=4, content="first"))

        assert db.summaries.get(second) is None

    def test_is_removed_with_its_branch(self, db: SqlAlchemy) -> None:
        branch = _branch(db)
        db.summaries.save(branch, Summary(covered=4, content="first"))

        db.remove_branch(branch)

        assert db.summaries.get(branch) is None

    def test_is_not_exported(self, db: SqlAlchemy) -> None:
        branch = _branch(db)
        db.summaries.save(branch, Summary(covered=4, content="first"))

        export = db.export_branch(branch)

        assert export is not None
        assert "first" not in str(export)

    def test_saving_for_an_unknown_branch_raises(self, db: SqlAlchemy) -> None:
        with pytest.raises(DatabaseError, match="the summary could not be stored"):
            db.summaries.save(uuid4(), Summary(covered=4, content="first"))

    def test_a_failed_read_raises(self, db: SqlAlchemy) -> None:
        with (
            patch(
                "sensai_server.memory.persistent_db.sql_alchemy.summary.Session.scalars", side_effect=SQLAlchemyError
            ),
            pytest.raises(DatabaseError, match="the summary could not be read"),
        ):
            db.summaries.get(uuid4())
