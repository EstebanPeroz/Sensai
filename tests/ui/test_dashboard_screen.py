import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import pytest
from textual.app import App
from textual.pilot import Pilot
from textual.widgets import DataTable, Input, Label, Static

from sensai.llm.responses import ChatMetrics
from sensai.monitoring.interaction import Interaction, InteractionStatus
from sensai.monitoring.metrics import Drift, Period, Report, build_report
from sensai.ui.event_queue import EventQueue
from sensai.ui.tui._dashboard import (
    DashboardScreen,
    format_drift,
    format_ms,
    format_number,
    format_rate,
    format_series,
)
from sensai.ui.tui._textual import ChatApp

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


def at(ago: timedelta, **overrides: object) -> Interaction:
    fields: dict = {"model": "qwen", "status": InteractionStatus.OK, "latency_ms": 1000.0, "created_at": NOW - ago}
    return Interaction(**(fields | overrides))


def busy_report() -> Report:
    slow_before = [at(timedelta(days=3), latency_ms=1000.0, persona="analyst") for _ in range(5)]
    slow_now = [
        at(
            timedelta(hours=1),
            latency_ms=2500.0,
            persona="analyst",
            metrics=ChatMetrics(prompt_tokens=10, completion_tokens=20, eval_duration_ms=1000.0),
        )
        for _ in range(5)
    ]
    failure = at(timedelta(minutes=5), status=InteractionStatus.ERROR, error="[RequestStatusError] status 404")
    return build_report([*slow_before, *slow_now, failure], Period.DAY, NOW)


class Host(App):
    """Bare app showing a dashboard screen."""

    def __init__(self, report: Report) -> None:
        super().__init__()
        self._report = report

    def on_mount(self) -> None:
        self.push_screen(DashboardScreen(self._report))


def run(app: App, scenario: Callable[[Pilot], Awaitable[None]]) -> None:
    async def main() -> None:
        async with app.run_test(size=(140, 60)) as pilot:
            await pilot.pause()
            await scenario(pilot)

    asyncio.run(main())


def text(pilot: Pilot, selector: str) -> str:
    return str(pilot.app.screen.query_one(selector, Static).render())


def all_text(pilot: Pilot) -> str:
    return "\n".join(str(label.render()) for label in pilot.app.screen.query(Label))


class TestFormatting:
    @pytest.mark.parametrize(
        ("value", "expected"), [(None, "—"), (0.0, "0 ms"), (850.4, "850 ms"), (1000.0, "1.0 s"), (3840.0, "3.8 s")]
    )
    def test_format_ms(self, value: float | None, expected: str) -> None:
        assert format_ms(value) == expected

    def test_format_rate(self) -> None:
        assert format_rate(None) == "—"
        assert format_rate(0.021) == "2.1 %"

    def test_format_number(self) -> None:
        assert format_number(None) == "—"
        assert format_number(31.44) == "31.4"

    def test_format_series_scales_from_zero_and_shows_gaps(self) -> None:
        assert format_series([0, 1, 4, 8]) == " ▁▄█"
        assert format_series([None, 2500.0, None]) == "·█·"
        assert format_series([None, None]) == "··"
        assert format_series([]) == ""

    def test_format_drift(self) -> None:
        drift = Drift("qwen", "analyst", baseline_p50_ms=1000.0, recent_p50_ms=1640.0)

        assert format_drift(drift) == "⚠ qwen / analyst: response time p50 1.0 s → 1.6 s (+64 %)"


class TestDashboardScreen:
    def test_shows_headline_figures(self) -> None:
        async def scenario(pilot: Pilot) -> None:
            labels = all_text(pilot)
            assert "Monitoring — last 24h" in labels
            assert "Calls\n6" in labels
            assert "Response time p50\n2.5 s" in labels
            assert "Errors\n16.7 %" in labels
            assert "Tokens/s\n20.0" in labels
            assert "Tokens in/out\n50/100" in labels
            assert not pilot.app.screen.query("#empty")

        run(Host(busy_report()), scenario)

    def test_shows_series_drift_groups_and_errors(self) -> None:
        async def scenario(pilot: Pilot) -> None:
            screen = pilot.app.screen
            assert len(text(pilot, "#calls-series")) == 24
            assert text(pilot, "#latency-series").count("·") == 23
            assert "⚠ qwen / analyst: response time p50 1.0 s → 2.5 s (+150 %)" in text(pilot, "#drifts")
            table = screen.query_one("#groups", DataTable)
            assert table.row_count == 2
            assert [str(cell) for cell in table.get_row_at(0)] == [
                "qwen",
                "analyst",
                "5",
                "2.5 s",
                "2.5 s",
                "0.0 %",
                "20.0",
            ]
            assert "[RequestStatusError] status 404" in text(pilot, "#errors")

        run(Host(busy_report()), scenario)

    def test_table_columns_fit_long_names_shown_as_plain_text(self) -> None:
        report = build_report([at(timedelta(minutes=1), model="llama3.1:70b-[q4]")], Period.DAY, NOW)

        async def scenario(pilot: Pilot) -> None:
            table = pilot.app.screen.query_one("#groups", DataTable)
            model_column = next(iter(table.columns.values()))
            assert str(table.get_row_at(0)[0]) == "llama3.1:70b-[q4]"
            assert model_column.content_width >= len("llama3.1:70b-[q4]")

        run(Host(report), scenario)

    def test_empty_report(self) -> None:
        async def scenario(pilot: Pilot) -> None:
            assert "No interaction logged over the last 7d." in text(pilot, "#empty")
            assert "No response time drift" in text(pilot, "#drifts")
            assert text(pilot, "#errors") == "None."
            assert pilot.app.screen.query_one("#groups", DataTable).row_count == 0

        run(Host(build_report([], Period.WEEK, NOW)), scenario)

    def test_error_text_is_not_read_as_markup(self) -> None:
        report = build_report(
            [at(timedelta(minutes=1), status=InteractionStatus.ERROR, error="bad [red]tag[/red]")], Period.DAY, NOW
        )

        async def scenario(pilot: Pilot) -> None:
            assert "bad [red]tag[/red]" in text(pilot, "#errors")

        run(Host(report), scenario)


class TestChatAppDashboard:
    @pytest.mark.parametrize("key", ["q", "escape"])
    def test_opens_over_the_chat_and_closes_back_to_it(self, key: str) -> None:
        app = ChatApp(EventQueue())

        async def scenario(pilot: Pilot) -> None:
            app.show_dashboard(busy_report())
            await pilot.pause()
            assert isinstance(app.screen, DashboardScreen)

            await pilot.press(key)
            await pilot.pause()
            assert not isinstance(app.screen, DashboardScreen)
            assert isinstance(app.focused, Input)

        run(app, scenario)
