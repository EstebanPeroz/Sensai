import math
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from sensai_protocol import (
    DashboardDrift,
    DashboardError,
    DashboardGroup,
    DashboardReport,
    DashboardSummary,
    Period,
)

from sensai_server.monitoring.interaction import Interaction, InteractionStatus

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

_MS_PER_S = 1000
_RECENT_ERRORS = 5

BASELINE_PERIODS = 7
"""Number of periods before the reported one that make up the reference for drift detection."""
DRIFT_THRESHOLD = 0.3
"""Relative increase of the median response time, over the reference, that counts as a drift."""
MIN_DRIFT_SAMPLES = 5
"""Successful calls needed on each side before comparing medians: fewer is noise."""


_DURATIONS = {Period.DAY: timedelta(days=1), Period.WEEK: timedelta(days=7), Period.MONTH: timedelta(days=30)}
"""Length of the window each period covers, ending now."""
_BUCKETS = {Period.DAY: timedelta(hours=1), Period.WEEK: timedelta(hours=6), Period.MONTH: timedelta(days=1)}
"""Width of one point of the time series: 24 hours, 28 quarters of a day, or 30 days."""


@dataclass(frozen=True)
class Summary:
    """Aggregated figures over a set of interactions.

    Latencies only count successful calls: an error fails fast and an interruption cuts the
    call short, so both would make the model look faster than it is.
    """

    calls: int
    errors: int
    interrupted: int
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    first_token_p50_ms: float | None
    tokens_per_s: float | None
    prompt_tokens: int
    completion_tokens: int

    @property
    def error_rate(self) -> float | None:
        """Share of the calls that failed, None without any call."""
        return self.errors / self.calls if self.calls else None

    @property
    def interrupted_rate(self) -> float | None:
        """Share of the calls the user cut short, None without any call."""
        return self.interrupted / self.calls if self.calls else None


@dataclass(frozen=True)
class GroupStats:
    """Summary of the interactions sharing a model and a persona."""

    model: str
    persona: str | None
    summary: Summary


@dataclass(frozen=True)
class Drift:
    """A model and persona pair whose median response time rose over the reference periods."""

    model: str
    persona: str | None
    baseline_p50_ms: float
    recent_p50_ms: float

    @property
    def change(self) -> float:
        """Relative increase of the median response time (0.5 for +50 %)."""
        return self.recent_p50_ms / self.baseline_p50_ms - 1


@dataclass(frozen=True)
class Report:
    """Everything a dashboard shows for one period, computed from the logged interactions."""

    period: Period
    start: datetime
    end: datetime
    summary: Summary
    calls_series: list[int]
    latency_series: list[float | None]
    groups: list[GroupStats]
    drifts: list[Drift]
    recent_errors: list[Interaction]


def history_start(period: Period, now: datetime) -> datetime:
    """Return the oldest date `build_report` looks at: the reference periods before the reported one."""
    return now - _DURATIONS[period] * (1 + BASELINE_PERIODS)


def build_report(
    interactions: Iterable[Interaction], period: Period, now: datetime, *, drift_threshold: float = DRIFT_THRESHOLD
) -> Report:
    """Aggregate the `interactions` made during the `period` ending at `now`.

    Those of the `BASELINE_PERIODS` before it are the reference for drift detection; older or
    later ones are ignored.
    """
    start = now - _DURATIONS[period]
    interactions = list(interactions)
    window = [i for i in interactions if start <= i.created_at <= now]
    baseline = [i for i in interactions if history_start(period, now) <= i.created_at < start]
    calls_series, latency_series = _series(window, start, period)
    return Report(
        period=period,
        start=start,
        end=now,
        summary=summarize(window),
        calls_series=calls_series,
        latency_series=latency_series,
        groups=_groups(window),
        drifts=_drifts(window, baseline, drift_threshold),
        recent_errors=sorted(
            (i for i in window if i.status is InteractionStatus.ERROR), key=lambda i: i.created_at, reverse=True
        )[:_RECENT_ERRORS],
    )


def summarize(interactions: list[Interaction]) -> Summary:
    """Aggregate `interactions` into call counts, response time percentiles, throughput and token totals."""
    ok = [i for i in interactions if i.status is InteractionStatus.OK]
    latencies = [i.latency_ms for i in ok]
    first_tokens = [i.first_token_ms for i in ok if i.first_token_ms is not None]
    return Summary(
        calls=len(interactions),
        errors=sum(i.status is InteractionStatus.ERROR for i in interactions),
        interrupted=sum(i.status is InteractionStatus.INTERRUPTED for i in interactions),
        latency_p50_ms=percentile(latencies, 0.5),
        latency_p95_ms=percentile(latencies, 0.95),
        first_token_p50_ms=percentile(first_tokens, 0.5),
        tokens_per_s=_tokens_per_s(ok),
        prompt_tokens=sum(i.metrics.prompt_tokens or 0 for i in interactions),
        completion_tokens=sum(i.metrics.completion_tokens or 0 for i in interactions),
    )


def percentile(values: list[float], q: float) -> float | None:
    """Return the `q` quantile (0 to 1) of `values`, linearly interpolated, or None if there is none."""
    if not values:
        return None
    ordered = sorted(values)
    rank = q * (len(ordered) - 1)
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _tokens_per_s(interactions: list[Interaction]) -> float | None:
    """Return the generation speed reported by the provider: generated tokens over generation time."""
    timed = [
        i
        for i in interactions
        if i.metrics.completion_tokens is not None and i.metrics.eval_duration_ms  # skips missing or zero durations
    ]
    seconds = sum(i.metrics.eval_duration_ms or 0 for i in timed) / _MS_PER_S
    if not seconds:
        return None
    return sum(i.metrics.completion_tokens or 0 for i in timed) / seconds


def _series(window: list[Interaction], start: datetime, period: Period) -> tuple[list[int], list[float | None]]:
    """Count the calls and the median latency of the successful ones (None without any) per bucket."""
    size = _DURATIONS[period] // _BUCKETS[period]
    buckets: list[list[Interaction]] = [[] for _ in range(size)]
    for interaction in window:
        buckets[min((interaction.created_at - start) // _BUCKETS[period], size - 1)].append(interaction)
    latencies = [
        percentile([i.latency_ms for i in bucket if i.status is InteractionStatus.OK], 0.5) for bucket in buckets
    ]
    return [len(bucket) for bucket in buckets], latencies


def _by_group(interactions: list[Interaction]) -> dict[tuple[str, str | None], list[Interaction]]:
    grouped: dict[tuple[str, str | None], list[Interaction]] = {}
    for interaction in interactions:
        grouped.setdefault((interaction.model, interaction.persona), []).append(interaction)
    return grouped


def _groups(window: list[Interaction]) -> list[GroupStats]:
    """Summarize each (model, persona) pair, busiest first."""
    stats = [GroupStats(model, persona, summarize(items)) for (model, persona), items in _by_group(window).items()]
    return sorted(stats, key=lambda group: (-group.summary.calls, group.model, group.persona or ""))


def _drifts(window: list[Interaction], baseline: list[Interaction], threshold: float) -> list[Drift]:
    """Find the (model, persona) pairs whose median latency rose by `threshold` or more, worst first."""
    reference = _by_group(baseline)
    drifts: list[Drift] = []
    for key, items in _by_group(window).items():
        recent = [i.latency_ms for i in items if i.status is InteractionStatus.OK]
        before = [i.latency_ms for i in reference.get(key, []) if i.status is InteractionStatus.OK]
        if min(len(recent), len(before)) < MIN_DRIFT_SAMPLES:
            continue
        drift = Drift(
            *key, baseline_p50_ms=percentile(before, 0.5) or 0.0, recent_p50_ms=percentile(recent, 0.5) or 0.0
        )
        if drift.baseline_p50_ms and drift.change >= threshold:
            drifts.append(drift)
    return sorted(drifts, key=lambda drift: drift.change, reverse=True)


def to_protocol(report: Report) -> DashboardReport:
    """Convert `report` to the wire format the client's dashboard reads."""
    return DashboardReport(
        period=report.period,
        start=report.start,
        end=report.end,
        summary=_summary_to_protocol(report.summary),
        calls_series=report.calls_series,
        latency_series=report.latency_series,
        groups=[
            DashboardGroup(model=group.model, persona=group.persona, summary=_summary_to_protocol(group.summary))
            for group in report.groups
        ],
        drifts=[
            DashboardDrift(
                model=drift.model,
                persona=drift.persona,
                baseline_p50_ms=drift.baseline_p50_ms,
                recent_p50_ms=drift.recent_p50_ms,
            )
            for drift in report.drifts
        ],
        recent_errors=[
            DashboardError(created_at=error.created_at, model=error.model, error=error.error)
            for error in report.recent_errors
        ],
        baseline_periods=BASELINE_PERIODS,
    )


def _summary_to_protocol(summary: Summary) -> DashboardSummary:
    return DashboardSummary(
        calls=summary.calls,
        errors=summary.errors,
        interrupted=summary.interrupted,
        latency_p50_ms=summary.latency_p50_ms,
        latency_p95_ms=summary.latency_p95_ms,
        first_token_p50_ms=summary.first_token_p50_ms,
        tokens_per_s=summary.tokens_per_s,
        prompt_tokens=summary.prompt_tokens,
        completion_tokens=summary.completion_tokens,
    )
