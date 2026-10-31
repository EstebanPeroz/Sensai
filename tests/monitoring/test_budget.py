from uuid import UUID, uuid4

import pytest

from sensai.config.settings import ContextSettings
from sensai.core.events import EventBus
from sensai.llm.responses import ChatMetrics
from sensai.monitoring.budget import ContextBudget, ContextUsage
from sensai.monitoring.interaction import Interaction, InteractionStatus

BRANCH = uuid4()
OTHER_BRANCH = uuid4()


def chat_call(
    prompt_tokens: int | None, completion_tokens: int | None = 0, *, branch_id: UUID | None = BRANCH
) -> Interaction:
    return Interaction(
        model="m",
        status=InteractionStatus.OK,
        latency_ms=1.0,
        branch_id=branch_id,
        metrics=ChatMetrics(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def attached(
    settings: ContextSettings, context_length: int | None = 32768
) -> tuple[EventBus, list[ContextUsage], list[str]]:
    bus, usages, lookups = EventBus(), [], []

    def lookup(model: str) -> int | None:
        lookups.append(model)
        return context_length

    ContextBudget(settings, lookup, usages.append).attach(bus)
    return bus, usages, lookups


class TestContextUsage:
    def test_ratio_is_the_share_of_the_budget_in_use(self) -> None:
        assert ContextUsage(used=2048, budget=8192, warning_threshold=0.8).ratio == 0.25

    def test_has_no_ratio_without_a_measure(self) -> None:
        usage = ContextUsage(used=None, budget=8192, warning_threshold=0.8)

        assert usage.ratio is None
        assert usage.near_limit is False

    def test_has_no_ratio_without_a_budget(self) -> None:
        usage = ContextUsage(used=2048, budget=None, warning_threshold=0.8)

        assert usage.ratio is None
        assert usage.near_limit is False

    @pytest.mark.parametrize(("used", "near_limit"), [(6553, False), (6554, True), (9000, True)])
    def test_is_near_the_limit_from_the_warning_threshold(self, used: int, *, near_limit: bool) -> None:
        assert ContextUsage(used=used, budget=8192, warning_threshold=0.8).near_limit is near_limit


class TestContextBudget:
    def test_measures_the_prompt_and_the_answer_of_the_last_chat_call(self) -> None:
        bus, usages, _ = attached(ContextSettings(budget=8192))

        bus.emit(chat_call(1000, 50))
        bus.emit(chat_call(1200, 300))

        assert usages == [ContextUsage(1050, 8192, 0.8), ContextUsage(1500, 8192, 0.8)]

    def test_ignores_calls_outside_of_a_branch(self) -> None:
        bus, usages, _ = attached(ContextSettings(budget=8192))

        bus.emit(chat_call(1000, branch_id=None))

        assert usages == []

    def test_ignores_calls_without_a_token_count(self) -> None:
        bus, usages, _ = attached(ContextSettings(budget=8192))

        bus.emit(chat_call(None))

        assert usages == []

    def test_counts_a_missing_answer_as_empty(self) -> None:
        bus, usages, _ = attached(ContextSettings(budget=8192))

        bus.emit(chat_call(1000, None))

        assert usages[0].used == 1000

    def test_the_budget_of_the_settings_wins_over_the_model(self) -> None:
        bus, usages, lookups = attached(ContextSettings(budget=8192), context_length=32768)

        bus.emit(chat_call(1000))

        assert usages[0].budget == 8192
        assert lookups == []

    def test_falls_back_on_the_context_length_of_the_model_looked_up_once(self) -> None:
        bus, usages, lookups = attached(ContextSettings(), context_length=32768)

        bus.emit(chat_call(1000))
        bus.emit(chat_call(2000))

        assert [usage.budget for usage in usages] == [32768, 32768]
        assert lookups == ["m"]

    def test_looks_the_context_length_up_again_while_the_model_cannot_tell(self) -> None:
        lengths = iter([None, 32768, 65536])
        bus, usages = EventBus(), []
        ContextBudget(ContextSettings(), lambda _: next(lengths), usages.append).attach(bus)

        for _ in range(3):
            bus.emit(chat_call(1000))

        assert [usage.budget for usage in usages] == [None, 32768, 32768]

    def test_a_conversation_not_started_takes_nothing(self) -> None:
        budget = ContextBudget(ContextSettings(budget=8192), lambda _: None, lambda _: None)

        assert budget.usage("m", None) == ContextUsage(0, 8192, 0.8)

    def test_a_branch_not_measured_yet_has_an_unknown_usage(self) -> None:
        budget = ContextBudget(ContextSettings(budget=8192), lambda _: None, lambda _: None)

        assert budget.usage("m", BRANCH) == ContextUsage(None, 8192, 0.8)

    def test_keeps_the_usage_of_each_branch(self) -> None:
        budget = ContextBudget(ContextSettings(budget=8192), lambda _: None, lambda _: None)
        bus = EventBus()
        budget.attach(bus)

        bus.emit(chat_call(1000, branch_id=BRANCH))
        bus.emit(chat_call(3000, branch_id=OTHER_BRANCH))

        assert budget.usage("m", BRANCH).used == 1000
        assert budget.usage("m", OTHER_BRANCH).used == 3000
