from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual.binding import Binding, BindingType
from textual.containers import HorizontalGroup, VerticalGroup, VerticalScroll
from textual.content import Content
from textual.screen import Screen
from textual.widgets import DataTable, Label, Static

import sensai.ui.tui._effect as fx
from sensai.monitoring.metrics import BASELINE_PERIODS, Period

if TYPE_CHECKING:
    from datetime import datetime

    from textual.app import ComposeResult

    from sensai.monitoring.metrics import Drift, Report

_MS_PER_S = 1000
_BUCKET_NAMES = {Period.DAY: "hour", Period.WEEK: "6 hours", Period.MONTH: "day"}
_NONE = "—"
_BARS = "▁▂▃▄▅▆▇█"
_GAP = "·"


def format_ms(value: float | None) -> str:
    """Render a duration: milliseconds under a second, seconds with one decimal above."""
    if value is None:
        return _NONE
    if value < _MS_PER_S:
        return f"{value:.0f} ms"
    return f"{value / _MS_PER_S:.1f} s"


def format_rate(value: float | None) -> str:
    """Render a share (0.021) as a percentage (2.1 %)."""
    return _NONE if value is None else f"{value * 100:.1f} %"


def format_number(value: float | None) -> str:
    """Render a measure with one decimal, or a dash when there is none."""
    return _NONE if value is None else f"{value:.1f}"


def format_time(value: datetime) -> str:
    """Render a stored (UTC) date in the local time of the machine."""
    return value.astimezone().strftime("%d/%m %H:%M")


def format_drift(drift: Drift) -> str:
    """Render a drift alert as one line: who, the two medians and the change."""
    who = f"{drift.model} / {drift.persona or _NONE}"
    medians = f"{format_ms(drift.baseline_p50_ms)} → {format_ms(drift.recent_p50_ms)}"
    return f"⚠ {who}: response time p50 {medians} (+{drift.change * 100:.0f} %)"


def format_series(values: list[float] | list[float | None]) -> str:
    """Render one bar per point, scaled from 0 to the highest value: blank for 0, a dot when there is no value.

    Unlike a `Sparkline`, which needs a number everywhere, a missing point stays visible as such.
    """
    peak = max((value for value in values if value is not None), default=0)
    return "".join(
        _GAP if value is None else " " if not value else _BARS[math.ceil(value / peak * len(_BARS)) - 1]
        for value in values
    )


class StatCard(VerticalGroup):
    """One headline figure, with its name above it."""

    def __init__(self, title: str, value: str) -> None:
        """Show `value` under `title`."""
        super().__init__(classes="card")
        self._title = title
        self._value = value

    def compose(self) -> ComposeResult:
        """Build the title and value labels."""
        yield Label(self._title, classes="card-title")
        yield Label(Content(self._value), classes="card-value")


class DashboardScreen(Screen):
    """Read-only view of a monitoring `Report`: headline figures, trends, drifts and recent errors."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape,q", "app.pop_screen", "Back")]

    CSS = """
    DashboardScreen {
        padding: 0 1;
    }
    DashboardScreen .title {
        width: 100%;
        margin-bottom: 1;
    }
    DashboardScreen #cards {
        height: auto;
    }
    DashboardScreen .card {
        width: 1fr;
        height: auto;
        border: round $primary;
        padding: 0 1;
    }
    DashboardScreen .card-title {
        color: $text-muted;
    }
    DashboardScreen .card-value {
        text-style: bold;
    }
    DashboardScreen .section {
        margin-top: 1;
    }
    DashboardScreen .series {
        color: $primary;
        margin-bottom: 1;
    }
    DashboardScreen DataTable {
        height: auto;
        max-height: 12;
    }
    DashboardScreen #drifts.alert {
        color: $warning;
    }
    DashboardScreen #errors.alert {
        color: $error;
    }
    DashboardScreen .hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    report: Report

    def __init__(self, report: Report) -> None:
        """Show `report`."""
        super().__init__()
        self.report = report

    def compose(self) -> ComposeResult:
        """Build the whole dashboard from the report."""
        report, summary = self.report, self.report.summary
        bucket = _BUCKET_NAMES[report.period]
        with VerticalScroll():
            yield Label(
                fx.bold(f"Monitoring — last {report.period.value}")
                + f"  ({format_time(report.start)} → {format_time(report.end)})",
                classes="title",
            )
            if summary.calls == 0:
                yield Static(f"No interaction logged over the last {report.period.value}.", id="empty")
            with HorizontalGroup(id="cards"):
                yield StatCard("Calls", str(summary.calls))
                yield StatCard("First token p50", format_ms(summary.first_token_p50_ms))
                yield StatCard("Response time p50", format_ms(summary.latency_p50_ms))
                yield StatCard("Response time p95", format_ms(summary.latency_p95_ms))
                yield StatCard("Errors", format_rate(summary.error_rate))
                yield StatCard("Interrupted", format_rate(summary.interrupted_rate))
                yield StatCard("Tokens/s", format_number(summary.tokens_per_s))
                yield StatCard("Tokens in/out", f"{summary.prompt_tokens}/{summary.completion_tokens}")
            yield Label(f"Calls per {bucket}", classes="section")
            yield Static(format_series(report.calls_series), id="calls-series", classes="series", markup=False)
            yield Label(f"Response time p50 per {bucket} ({_GAP} no successful call)", classes="section")
            yield Static(format_series(report.latency_series), id="latency-series", classes="series", markup=False)
            yield self._drifts()
            yield Label("By model / persona", classes="section")
            yield DataTable(id="groups", cursor_type="row", zebra_stripes=True)
            yield Label("Recent errors", classes="section")
            yield self._errors()
            yield Label("q / esc: back  ·  /dashboard 24h | 7d | 30d: change the period", classes="hint")

    def on_mount(self) -> None:
        """Fill the per model / persona table, which needs to be mounted first."""
        table = self.query_one("#groups", DataTable)
        # rich `Text` rather than `Content`: DataTable sizes its columns from `Text`, not from `Content`.
        table.add_columns("Model", "Persona", "Calls", "p50", "p95", "Errors", "Tokens/s")
        for group in self.report.groups:
            stats = group.summary
            table.add_row(
                Text(group.model),
                Text(group.persona or _NONE),
                str(stats.calls),
                format_ms(stats.latency_p50_ms),
                format_ms(stats.latency_p95_ms),
                format_rate(stats.error_rate),
                format_number(stats.tokens_per_s),
            )

    # Model names, personas and error messages come from the database: they are shown as plain
    # `Content`, never parsed as markup, where a "[RequestStatusError]" would vanish as a tag.

    def _drifts(self) -> Static:
        reference = f"the previous {BASELINE_PERIODS} periods of {self.report.period.value}"
        if not self.report.drifts:
            return Static(f"No response time drift against {reference}.", id="drifts", classes="section")
        lines = [f"Response time drift against {reference}:", *(format_drift(drift) for drift in self.report.drifts)]
        return Static(Content("\n".join(lines)), id="drifts", classes="section alert")

    def _errors(self) -> Static:
        if not self.report.recent_errors:
            return Static("None.", id="errors")
        lines = [
            f"{format_time(i.created_at)}  {i.model}  {i.error or 'unknown error'}" for i in self.report.recent_errors
        ]
        return Static(Content("\n".join(lines)), id="errors", classes="alert")
