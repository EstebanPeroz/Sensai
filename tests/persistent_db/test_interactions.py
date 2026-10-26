from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from sensai.llm.responses import ChatMetrics
from sensai.memory.persistent_db.error import DatabaseError
from sensai.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai.monitoring.interaction import Interaction, InteractionStatus


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
        newer = make_interaction(branch_id=db.create_branch(), created_at=now)
        older = make_interaction(status=InteractionStatus.ERROR, error="refused", created_at=now - timedelta(minutes=5))

        db.interactions.add(newer)
        db.interactions.add(older)

        assert db.interactions.get_all() == [older, newer]

    def test_unknown_branch_is_dropped(self, db: SqlAlchemy) -> None:
        db.interactions.add(make_interaction(branch_id=uuid4()))

        assert db.interactions.get_all()[0].branch_id is None

    def test_kept_when_its_branch_is_removed(self, db: SqlAlchemy) -> None:
        branch = db.create_branch()
        assert branch is not None
        db.interactions.add(make_interaction(branch_id=branch))

        db.remove_branch(branch)

        assert [i.branch_id for i in db.interactions.get_all()] == [None]

    def test_raises_database_error_when_commit_fails(self, db: SqlAlchemy) -> None:
        with patch.object(Session, "commit", side_effect=SQLAlchemyError), pytest.raises(DatabaseError):
            db.interactions.add(make_interaction())
