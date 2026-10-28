from datetime import UTC, datetime, timedelta

import pytest

from sensai.llm.responses import ChatMetrics
from sensai.monitoring.interaction import Interaction, InteractionStatus
from sensai.monitoring.metrics import (
    BASELINE_PERIODS,
    MIN_DRIFT_SAMPLES,
    Period,
    Summary,
    build_report,
    history_start,
    percentile,
    summarize,
)

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


def at(ago: timedelta, **overrides: object) -> Interaction:
    """An interaction made `ago` before NOW: successful, 1 s long, on the qwen model, unless overridden."""
    fields: dict = {"model": "qwen", "status": InteractionStatus.OK, "latency_ms": 1000.0, "created_at": NOW - ago}
    return Interaction(**(fields | overrides))


class TestPeriod:
    @pytest.mark.parametrize(("period", "points"), [(Period.DAY, 24), (Period.WEEK, 28), (Period.MONTH, 30)])
    def test_number_of_points(self, period: Period, points: int) -> None:
        assert period.duration // period.bucket == points

    def test_parses_its_command_argument(self) -> None:
        assert Period("7d") is Period.WEEK


class TestPercentile:
    def test_none_without_values(self) -> None:
        assert percentile([], 0.5) is None

    def test_single_value(self) -> None:
        assert percentile([42.0], 0.95) == 42.0

    def test_median_of_an_even_count_is_interpolated(self) -> None:
        assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.5

    def test_p95(self) -> None:
        assert percentile([float(v) for v in range(1, 101)], 0.95) == pytest.approx(95.05)


class TestSummarize:
    def test_empty(self) -> None:
        summary = summarize([])

        assert summary == Summary(
            calls=0,
            errors=0,
            interrupted=0,
            latency_p50_ms=None,
            latency_p95_ms=None,
            first_token_p50_ms=None,
            tokens_per_s=None,
            prompt_tokens=0,
            completion_tokens=0,
        )
        assert summary.error_rate is None
        assert summary.interrupted_rate is None

    def test_counts_statuses_and_rates(self) -> None:
        summary = summarize(
            [
                at(timedelta(0)),
                at(timedelta(0)),
                at(timedelta(0), status=InteractionStatus.ERROR),
                at(timedelta(0), status=InteractionStatus.INTERRUPTED),
            ]
        )

        assert (summary.calls, summary.errors, summary.interrupted) == (4, 1, 1)
        assert summary.error_rate == 0.25
        assert summary.interrupted_rate == 0.25

    def test_latency_only_counts_successful_calls(self) -> None:
        summary = summarize(
            [
                at(timedelta(0), latency_ms=1000.0),
                at(timedelta(0), latency_ms=3000.0),
                at(timedelta(0), latency_ms=2.0, status=InteractionStatus.ERROR),
                at(timedelta(0), latency_ms=10.0, status=InteractionStatus.INTERRUPTED),
            ]
        )

        assert summary.latency_p50_ms == 2000.0
        assert summary.latency_p95_ms == pytest.approx(2900.0)

    def test_first_token_median_skips_missing_values(self) -> None:
        with_ttft = Interaction(model="m", status=InteractionStatus.OK, latency_ms=900.0, first_token_ms=300.0)
        without_ttft = Interaction(model="m", status=InteractionStatus.OK, latency_ms=900.0)

        assert summarize([with_ttft, without_ttft]).first_token_p50_ms == 300.0

    def test_tokens_per_second_and_totals(self) -> None:
        summary = summarize(
            [
                at(timedelta(0), metrics=ChatMetrics(prompt_tokens=10, completion_tokens=30, eval_duration_ms=1000.0)),
                at(timedelta(0), metrics=ChatMetrics(prompt_tokens=20, completion_tokens=10, eval_duration_ms=1000.0)),
                at(timedelta(0), metrics=ChatMetrics(prompt_tokens=5, completion_tokens=99, eval_duration_ms=0.0)),
                at(timedelta(0), status=InteractionStatus.ERROR),
            ]
        )

        assert summary.tokens_per_s == 20.0
        assert summary.prompt_tokens == 35
        assert summary.completion_tokens == 139

    def test_tokens_per_second_is_none_without_timing(self) -> None:
        assert summarize([at(timedelta(0), metrics=ChatMetrics(completion_tokens=5))]).tokens_per_s is None


class TestBuildReport:
    def test_only_keeps_the_period(self) -> None:
        report = build_report(
            [at(timedelta(hours=25)), at(timedelta(hours=23)), at(timedelta(minutes=-5))], Period.DAY, NOW
        )

        assert report.summary.calls == 1
        assert report.start == NOW - timedelta(days=1)
        assert report.end == NOW

    def test_empty_period(self) -> None:
        report = build_report([], Period.DAY, NOW)

        assert report.summary.calls == 0
        assert report.calls_series == [0] * 24
        assert report.latency_series == [None] * 24
        assert report.groups == []
        assert report.recent_errors == []

    def test_series_buckets_calls_by_hour(self) -> None:
        report = build_report(
            [
                at(timedelta(hours=23, minutes=30), latency_ms=100.0),
                at(timedelta(minutes=30), latency_ms=200.0),
                at(timedelta(minutes=10), latency_ms=400.0),
                at(timedelta(0), latency_ms=5.0, status=InteractionStatus.ERROR),
            ],
            Period.DAY,
            NOW,
        )

        assert report.calls_series == [1] + [0] * 22 + [3]
        assert report.latency_series == [100.0] + [None] * 22 + [300.0]

    def test_week_series_has_six_hour_points(self) -> None:
        report = build_report([at(timedelta(days=6, hours=20)), at(timedelta(hours=1))], Period.WEEK, NOW)

        assert len(report.calls_series) == 28
        assert report.calls_series[0] == 1
        assert report.calls_series[-1] == 1

    def test_groups_by_model_and_persona_busiest_first(self) -> None:
        report = build_report(
            [
                at(timedelta(0), model="qwen", persona="analyst"),
                at(timedelta(0), model="llama"),
                at(timedelta(0), model="llama"),
                at(timedelta(0), model="qwen"),
            ],
            Period.DAY,
            NOW,
        )

        assert [(g.model, g.persona, g.summary.calls) for g in report.groups] == [
            ("llama", None, 2),
            ("qwen", None, 1),
            ("qwen", "analyst", 1),
        ]

    def test_recent_errors_newest_first_and_capped(self) -> None:
        errors = [
            at(timedelta(minutes=m), status=InteractionStatus.ERROR, error=f"error {m}") for m in (1, 2, 3, 4, 5, 6)
        ]

        report = build_report([at(timedelta(0)), *reversed(errors)], Period.DAY, NOW)

        assert [i.error for i in report.recent_errors] == ["error 1", "error 2", "error 3", "error 4", "error 5"]


def calls(count: int, ago: timedelta, latency_ms: float, **overrides: object) -> list[Interaction]:
    return [at(ago, latency_ms=latency_ms, **overrides) for _ in range(count)]


BEFORE = timedelta(days=3)  # inside the reference periods of a 24h report
RECENT = timedelta(hours=1)


class TestHistoryStart:
    def test_covers_the_reference_periods(self) -> None:
        assert history_start(Period.DAY, NOW) == NOW - timedelta(days=1 + BASELINE_PERIODS)


class TestDrift:
    def test_flags_a_slowdown_above_the_threshold(self) -> None:
        report = build_report(
            calls(5, BEFORE, 1000.0, persona="analyst") + calls(5, RECENT, 2000.0, persona="analyst"), Period.DAY, NOW
        )

        assert len(report.drifts) == 1
        drift = report.drifts[0]
        assert (drift.model, drift.persona) == ("qwen", "analyst")
        assert (drift.baseline_p50_ms, drift.recent_p50_ms) == (1000.0, 2000.0)
        assert drift.change == 1.0

    def test_ignores_small_changes_and_speedups(self) -> None:
        report = build_report(
            calls(5, BEFORE, 1000.0, model="a")
            + calls(5, RECENT, 1200.0, model="a")
            + calls(5, BEFORE, 1000.0, model="b")
            + calls(5, RECENT, 300.0, model="b"),
            Period.DAY,
            NOW,
        )

        assert report.drifts == []

    def test_threshold_is_configurable(self) -> None:
        interactions = calls(5, BEFORE, 1000.0) + calls(5, RECENT, 1200.0)

        assert len(build_report(interactions, Period.DAY, NOW, drift_threshold=0.1).drifts) == 1

    def test_needs_enough_samples_on_both_sides(self) -> None:
        few = MIN_DRIFT_SAMPLES - 1
        report = build_report(calls(few, BEFORE, 1000.0) + calls(5, RECENT, 5000.0), Period.DAY, NOW)

        assert report.drifts == []

    def test_reference_ignores_failed_calls_and_older_history(self) -> None:
        report = build_report(
            calls(5, BEFORE, 1000.0)
            + calls(20, BEFORE, 1.0, status=InteractionStatus.ERROR)
            + calls(20, timedelta(days=1 + BASELINE_PERIODS, hours=1), 1.0)
            + calls(5, RECENT, 1500.0),
            Period.DAY,
            NOW,
        )

        assert [d.baseline_p50_ms for d in report.drifts] == [1000.0]

    def test_pairs_are_compared_separately_worst_first(self) -> None:
        report = build_report(
            calls(5, BEFORE, 1000.0, persona="analyst")
            + calls(5, RECENT, 1500.0, persona="analyst")
            + calls(5, BEFORE, 1000.0)
            + calls(5, RECENT, 3000.0),
            Period.DAY,
            NOW,
        )

        assert [(d.persona, d.change) for d in report.drifts] == [(None, 2.0), ("analyst", 0.5)]
