import pytest

from sensai.monitoring.budget import ContextUsage
from sensai.ui.tui._textual import _format_usage


class TestFormatUsage:
    @pytest.mark.parametrize(
        ("usage", "expected"),
        [
            (ContextUsage(0, 8192, 0.8), "context: 0 / 8.2k (0%)"),
            (ContextUsage(2048, 8192, 0.8), "context: 2.0k / 8.2k (25%)"),
            (ContextUsage(950, None, 0.8), "context: 950"),
        ],
    )
    def test_shows_the_usage_against_the_budget(self, usage: ContextUsage, expected: str) -> None:
        assert _format_usage(usage) == expected

    def test_turns_red_near_the_limit(self) -> None:
        assert _format_usage(ContextUsage(7000, 8192, 0.8)) == "[red]context: 7.0k / 8.2k (85%)[/red]"
