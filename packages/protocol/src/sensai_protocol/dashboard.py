from datetime import datetime  # noqa: TC003 - pydantic resolves annotations at runtime
from enum import StrEnum

from sensai_protocol._base import ProtocolModel


class Period(StrEnum):
    """Time window a dashboard report covers, ending now."""

    DAY = "24h"
    WEEK = "7d"
    MONTH = "30d"


class DashboardSummary(ProtocolModel):
    """Aggregated figures over a set of model calls. Latencies only count successful calls."""

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


class DashboardGroup(ProtocolModel):
    """Summary of the calls sharing a model and a persona."""

    model: str
    persona: str | None
    summary: DashboardSummary


class DashboardDrift(ProtocolModel):
    """A model and persona pair whose median response time rose over the reference periods."""

    model: str
    persona: str | None
    baseline_p50_ms: float
    recent_p50_ms: float

    @property
    def change(self) -> float:
        """Relative increase of the median response time (0.5 for +50 %)."""
        return self.recent_p50_ms / self.baseline_p50_ms - 1


class DashboardError(ProtocolModel):
    """A recent failed call to the model."""

    created_at: datetime
    model: str
    error: str | None


class DashboardReport(ProtocolModel):
    """Everything a dashboard shows for one period.

    `baseline_periods` is the number of periods before this one that drifts are measured against.
    """

    period: Period
    start: datetime
    end: datetime
    summary: DashboardSummary
    calls_series: list[int]
    latency_series: list[float | None]
    groups: list[DashboardGroup]
    drifts: list[DashboardDrift]
    recent_errors: list[DashboardError]
    baseline_periods: int
