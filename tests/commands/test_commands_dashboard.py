from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from sensai.commands.dashboard import Dashboard
from sensai.memory.persistent_db.error import DatabaseError
from sensai.monitoring.interaction import Interaction, InteractionStatus
from sensai.monitoring.interaction_repository import InteractionRepository
from sensai.monitoring.metrics import BASELINE_PERIODS, Period
from tests.commands.fakes import FakeUI

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


@pytest.fixture
def db() -> MagicMock:
    db = MagicMock(spec=InteractionRepository)
    db.get_all.return_value = [
        Interaction(model="qwen", status=InteractionStatus.OK, latency_ms=800.0, created_at=NOW - timedelta(hours=1)),
        Interaction(model="qwen", status=InteractionStatus.OK, latency_ms=800.0, created_at=NOW - timedelta(days=3)),
    ]
    return db


def dashboard(db: MagicMock) -> Dashboard:
    return Dashboard(db, clock=lambda: NOW)


class TestDashboard:
    def test_defaults_to_the_last_24h(self, fake_ui: FakeUI, db: MagicMock) -> None:
        dashboard(db).execute()

        db.get_all.assert_called_once_with(since=NOW - timedelta(days=1 + BASELINE_PERIODS), with_content=False)
        assert len(fake_ui.dashboards) == 1
        report = fake_ui.dashboards[0]
        assert report.period is Period.DAY
        assert report.end == NOW
        assert report.summary.calls == 1
        assert fake_ui.messages == []

    @pytest.mark.parametrize(("arg", "period"), [("24h", Period.DAY), ("7d", Period.WEEK), ("30d", Period.MONTH)])
    def test_period_argument(self, fake_ui: FakeUI, db: MagicMock, arg: str, period: Period) -> None:
        dashboard(db).execute(arg)

        assert fake_ui.dashboards[0].period is period
        db.get_all.assert_called_once_with(since=NOW - period.duration * (1 + BASELINE_PERIODS), with_content=False)

    def test_unknown_period(self, fake_ui: FakeUI, db: MagicMock) -> None:
        dashboard(db).execute("1y")

        assert fake_ui.messages == ["Unknown period '1y'. Usage: /dashboard [24h|7d|30d]"]
        assert fake_ui.dashboards == []
        db.get_all.assert_not_called()

    def test_too_many_arguments(self, fake_ui: FakeUI, db: MagicMock) -> None:
        dashboard(db).execute("24h", "7d")

        assert fake_ui.messages == ["Invalid number of arguments. Usage: /dashboard [24h|7d|30d]"]
        assert fake_ui.dashboards == []

    def test_database_error_is_reported(self, fake_ui: FakeUI, db: MagicMock) -> None:
        db.get_all.side_effect = DatabaseError("unreachable")

        dashboard(db).execute()

        assert fake_ui.messages == ["[DatabaseError]unreachable"]
        assert fake_ui.dashboards == []

    def test_completes_the_periods(self, db: MagicMock) -> None:
        assert dashboard(db).completions([]) == ["24h", "7d", "30d"]

    def test_uses_the_current_time_by_default(self, fake_ui: FakeUI, db: MagicMock) -> None:
        before = datetime.now(tz=UTC)

        Dashboard(db).execute()

        assert before <= fake_ui.dashboards[0].end <= datetime.now(tz=UTC)
