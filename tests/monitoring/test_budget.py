import pytest

from sensai.monitoring.budget import ContextUsage


class TestContextUsage:
    def test_ratio_is_the_share_of_the_budget_in_use(self) -> None:
        assert ContextUsage(used=2048, budget=8192, warning_threshold=0.8).ratio == 0.25

    def test_has_no_ratio_without_a_budget(self) -> None:
        usage = ContextUsage(used=2048, budget=None, warning_threshold=0.8)

        assert usage.ratio is None
        assert usage.near_limit is False

    @pytest.mark.parametrize(("used", "near_limit"), [(6553, False), (6554, True), (9000, True)])
    def test_is_near_the_limit_from_the_warning_threshold(self, used: int, *, near_limit: bool) -> None:
        assert ContextUsage(used=used, budget=8192, warning_threshold=0.8).near_limit is near_limit
