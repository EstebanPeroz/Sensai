from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError

from sensai.memory.persistent_db.error import DatabaseError
from sensai.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai.memory.summary import Summary


class TestSummaries:
    def test_a_branch_starts_without_summary(self, db: SqlAlchemy) -> None:
        branch = db.create_branch()
        assert branch is not None

        assert db.summaries.get(branch) is None

    def test_round_trip(self, db: SqlAlchemy) -> None:
        branch = db.create_branch()
        assert branch is not None

        db.summaries.save(branch, Summary(covered=4, content="The user is Ada, she wants a budget of 300 EUR."))

        assert db.summaries.get(branch) == Summary(covered=4, content="The user is Ada, she wants a budget of 300 EUR.")

    def test_a_new_summary_replaces_the_previous_one(self, db: SqlAlchemy) -> None:
        branch = db.create_branch()
        assert branch is not None

        db.summaries.save(branch, Summary(covered=4, content="first"))
        db.summaries.save(branch, Summary(covered=10, content="second"))

        assert db.summaries.get(branch) == Summary(covered=10, content="second")

    def test_each_branch_has_its_own(self, db: SqlAlchemy) -> None:
        first, second = db.create_branch(), db.create_branch()
        assert first is not None
        assert second is not None

        db.summaries.save(first, Summary(covered=4, content="first"))

        assert db.summaries.get(second) is None

    def test_is_removed_with_its_branch(self, db: SqlAlchemy) -> None:
        branch = db.create_branch()
        assert branch is not None
        db.summaries.save(branch, Summary(covered=4, content="first"))

        db.remove_branch(branch)

        assert db.summaries.get(branch) is None

    def test_saving_for_an_unknown_branch_raises(self, db: SqlAlchemy) -> None:
        with pytest.raises(DatabaseError, match="the summary could not be stored"):
            db.summaries.save(uuid4(), Summary(covered=4, content="first"))

    def test_a_failed_read_raises(self, db: SqlAlchemy) -> None:
        with (
            patch("sensai.memory.persistent_db.sql_alchemy.summary.Session.scalars", side_effect=SQLAlchemyError),
            pytest.raises(DatabaseError, match="the summary could not be read"),
        ):
            db.summaries.get(uuid4())
