from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from sensai_server.llm.responses import ChatMetrics
from sensai_server.memory.persistent_db.error import DatabaseError
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai_server.monitoring.interaction import Interaction, InteractionStatus
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from tests.server.conftest import MODEL, db_url


@pytest.fixture
def db(tmp_path: Path) -> SqlAlchemy:
    return SqlAlchemy(db_url(tmp_path))


def make_interaction(**overrides: object) -> Interaction:
    fields: dict = {
        "model": "qwen3:1.7b",
        "status": InteractionStatus.OK,
        "latency_ms": 1234.5,
        "persona": "analyst",
        "request": [{"role": "user", "content": "hi"}],
        "response": "hello",
        "thinking": "greet back",
        "tool_calls": [{"function": {"name": "web_search", "arguments": {"q": "x"}}}],
        "first_token_ms": 120.0,
        "metrics": ChatMetrics(prompt_tokens=10, completion_tokens=2, total_duration_ms=1200.0, eval_duration_ms=900.0),
    }
    return Interaction(**(fields | overrides))


class TestInteractions:
    def test_round_trip_oldest_first(self, db: SqlAlchemy) -> None:
        now = datetime.now(tz=UTC)
        newer = make_interaction(branch_id=db.create_branch(model=MODEL), created_at=now)
        older = make_interaction(status=InteractionStatus.ERROR, error="refused", created_at=now - timedelta(minutes=5))

        db.interactions.add(newer)
        db.interactions.add(older)

        assert db.interactions.get_all() == [older, newer]

    def test_since_keeps_newer_interactions_whatever_the_timezone(self, db: SqlAlchemy) -> None:
        paris, ten = timezone(timedelta(hours=2)), datetime(2026, 10, 14, 10, 0, tzinfo=UTC)
        old, new = (
            make_interaction(created_at=ten),
            make_interaction(created_at=(ten + timedelta(hours=1)).astimezone(paris)),
        )
        db.interactions.add(old)
        db.interactions.add(new)

        assert [i.created_at for i in db.interactions.get_all()] == [ten, ten + timedelta(hours=1)]
        assert [i.id for i in db.interactions.get_all(since=(ten + timedelta(minutes=30)).astimezone(paris))] == [
            new.id
        ]
        assert [i.id for i in db.interactions.get_all(since=ten)] == [old.id, new.id]

    def test_without_content_keeps_only_the_metrics(self, db: SqlAlchemy) -> None:
        stored = make_interaction(status=InteractionStatus.ERROR, error="refused")
        db.interactions.add(stored)

        [light] = db.interactions.get_all(with_content=False)

        assert light == replace(stored, request=None, response=None, thinking=None, tool_calls=[])

    def test_unknown_branch_is_dropped(self, db: SqlAlchemy) -> None:
        db.interactions.add(make_interaction(branch_id=uuid4()))

        assert db.interactions.get_all()[0].branch_id is None

    def test_kept_when_its_branch_is_removed(self, db: SqlAlchemy) -> None:
        branch = db.create_branch(model=MODEL)
        assert branch is not None
        db.interactions.add(make_interaction(branch_id=branch))

        db.remove_branch(branch)

        assert [i.branch_id for i in db.interactions.get_all()] == [None]

    def test_raises_database_error_when_commit_fails(self, db: SqlAlchemy) -> None:
        with patch.object(Session, "commit", side_effect=SQLAlchemyError), pytest.raises(DatabaseError):
            db.interactions.add(make_interaction())

    def test_raises_database_error_when_reading_fails(self, db: SqlAlchemy) -> None:
        with patch.object(Session, "scalars", side_effect=SQLAlchemyError), pytest.raises(DatabaseError):
            db.interactions.get_all()
