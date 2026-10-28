from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import pytest
from sensai_protocol import (
    DashboardDrift,
    DashboardError,
    DashboardGroup,
    DashboardReport,
    DashboardSummary,
    Period,
)
from sensai_tui.ui._dashboard import (
    DashboardScreen,
    format_drift,
    format_ms,
    format_number,
    format_rate,
    format_series,
)
from textual.app import App
from textual.pilot import Pilot
from textual.widgets import DataTable, Label, Static

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


def summary(**overrides: object) -> DashboardSummary:
    fields: dict = {
        "calls": 0,
        "errors": 0,
        "interrupted": 0,
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "first_token_p50_ms": None,
        "tokens_per_s": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    return DashboardSummary(**(fields | overrides))


def report(**overrides: object) -> DashboardReport:
    fields: dict = {
        "period": Period.DAY,
        "start": datetime(2026, 10, 13, 12, 0, tzinfo=UTC),
        "end": NOW,
        "summary": summary(),
        "calls_series": [0] * 24,
        "latency_series": [None] * 24,
        "groups": [],
        "drifts": [],
        "recent_errors": [],
        "baseline_periods": 7,
    }
    return DashboardReport(**(fields | overrides))


def busy_report() -> DashboardReport:
    analyst = summary(
        calls=5,
        latency_p50_ms=2500.0,
        latency_p95_ms=2500.0,
        first_token_p50_ms=300.0,
        tokens_per_s=20.0,
        prompt_tokens=50,
        completion_tokens=100,
    )
    return report(
        summary=summary(
            calls=6,
            errors=1,
            latency_p50_ms=2500.0,
            latency_p95_ms=2500.0,
            first_token_p50_ms=300.0,
            tokens_per_s=20.0,
            prompt_tokens=50,
            completion_tokens=100,
        ),
        calls_series=[0] * 22 + [5, 1],
        latency_series=[None] * 22 + [2500.0, None],
        groups=[
            DashboardGroup(model="qwen", persona="analyst", summary=analyst),
            DashboardGroup(model="qwen", persona=None, summary=summary(calls=1, errors=1)),
        ],
        drifts=[DashboardDrift(model="qwen", persona="analyst", baseline_p50_ms=1000.0, recent_p50_ms=2500.0)],
        recent_errors=[DashboardError(created_at=NOW, model="qwen", error="[RequestStatusError] status 404")],
    )


class Host(App):
    """Bare app showing a dashboard screen."""

    def __init__(self, report: DashboardReport) -> None:
        super().__init__()
        self._report = report

    def on_mount(self) -> None:
        self.push_screen(DashboardScreen(self._report))


async def run(app: App, scenario: Callable[[Pilot], Awaitable[None]]) -> None:
    async with app.run_test(size=(140, 60)) as pilot:
        await pilot.pause()
        await scenario(pilot)


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
        drift = DashboardDrift(model="qwen", persona="analyst", baseline_p50_ms=1000.0, recent_p50_ms=1640.0)

        assert format_drift(drift) == "⚠ qwen / analyst: response time p50 1.0 s → 1.6 s (+64 %)"


class TestDashboardScreen:
    async def test_shows_headline_figures(self) -> None:
        async def scenario(pilot: Pilot) -> None:
            labels = all_text(pilot)
            assert "Monitoring — last 24h" in labels
            assert "Calls\n6" in labels
            assert "Response time p50\n2.5 s" in labels
            assert "Errors\n16.7 %" in labels
            assert "Tokens/s\n20.0" in labels
            assert "Tokens in/out\n50/100" in labels
            assert not pilot.app.screen.query("#empty")

        await run(Host(busy_report()), scenario)

    async def test_shows_series_drift_groups_and_errors(self) -> None:
        async def scenario(pilot: Pilot) -> None:
            assert len(text(pilot, "#calls-series")) == 24
            assert text(pilot, "#latency-series").count("·") == 23
            drifts = text(pilot, "#drifts")
            assert "against the previous 7 periods of 24h" in drifts
            assert "⚠ qwen / analyst: response time p50 1.0 s → 2.5 s (+150 %)" in drifts
            table = pilot.app.screen.query_one("#groups", DataTable)
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

        await run(Host(busy_report()), scenario)

    async def test_table_columns_fit_long_names_shown_as_plain_text(self) -> None:
        name = "llama3.1:70b-[q4]"
        groups = [DashboardGroup(model=name, persona=None, summary=summary(calls=1))]

        async def scenario(pilot: Pilot) -> None:
            table = pilot.app.screen.query_one("#groups", DataTable)
            model_column = next(iter(table.columns.values()))
            assert str(table.get_row_at(0)[0]) == name
            assert model_column.content_width >= len(name)

        await run(Host(report(summary=summary(calls=1), groups=groups)), scenario)

    async def test_empty_report(self) -> None:
        async def scenario(pilot: Pilot) -> None:
            assert "No interaction logged over the last 7d." in text(pilot, "#empty")
            assert "No response time drift" in text(pilot, "#drifts")
            assert text(pilot, "#errors") == "None."
            assert pilot.app.screen.query_one("#groups", DataTable).row_count == 0

        week = report(period=Period.WEEK, calls_series=[0] * 28, latency_series=[None] * 28)
        await run(Host(week), scenario)

    async def test_error_text_is_not_read_as_markup(self) -> None:
        errors = [DashboardError(created_at=NOW, model="qwen", error="bad [red]tag[/red]")]

        async def scenario(pilot: Pilot) -> None:
            assert "bad [red]tag[/red]" in text(pilot, "#errors")

        await run(Host(report(summary=summary(calls=1, errors=1), recent_errors=errors)), scenario)
