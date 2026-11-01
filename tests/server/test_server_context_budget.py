import threading
from uuid import UUID, uuid4

import pytest
from sensai_server.config.settings import ContextSettings
from sensai_server.events import EventBus
from sensai_server.llm.responses import ChatMetrics
from sensai_server.monitoring.budget import ContextBudget, ContextUsage
from sensai_server.monitoring.interaction import Interaction, InteractionStatus

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
) -> tuple[EventBus, ContextBudget, list[str]]:
    bus, lookups = EventBus(), []

    def lookup(model: str) -> int | None:
        lookups.append(model)
        return context_length

    budget = ContextBudget(settings, lookup)
    budget.attach(bus)
    return bus, budget, lookups


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
    async def test_measures_the_prompt_and_the_answer_of_the_last_chat_call(self) -> None:
        bus, budget, _ = attached(ContextSettings(budget=8192))

        bus.emit(chat_call(1000, 50))
        assert await budget.usage("m", BRANCH) == ContextUsage(1050, 8192, 0.8)

        bus.emit(chat_call(1200, 300))
        assert await budget.usage("m", BRANCH) == ContextUsage(1500, 8192, 0.8)

    async def test_ignores_calls_outside_of_a_branch(self) -> None:
        bus, budget, _ = attached(ContextSettings(budget=8192))

        bus.emit(chat_call(1000, branch_id=None))

        assert (await budget.usage("m", BRANCH)).used is None

    async def test_ignores_calls_without_a_token_count(self) -> None:
        bus, budget, _ = attached(ContextSettings(budget=8192))

        bus.emit(chat_call(None))

        assert (await budget.usage("m", BRANCH)).used is None

    async def test_counts_a_missing_answer_as_empty(self) -> None:
        bus, budget, _ = attached(ContextSettings(budget=8192))

        bus.emit(chat_call(1000, None))

        assert (await budget.usage("m", BRANCH)).used == 1000

    async def test_ignores_other_events(self) -> None:
        bus, budget, _ = attached(ContextSettings(budget=8192))

        bus.emit("not an interaction")

        assert (await budget.usage("m", BRANCH)).used is None

    async def test_the_budget_of_the_settings_wins_over_the_model(self) -> None:
        bus, budget, lookups = attached(ContextSettings(budget=8192), context_length=32768)

        bus.emit(chat_call(1000))

        assert (await budget.usage("m", BRANCH)).budget == 8192
        assert lookups == []

    async def test_falls_back_on_the_context_length_of_the_model_looked_up_once(self) -> None:
        _, budget, lookups = attached(ContextSettings(), context_length=32768)

        assert (await budget.usage("m", BRANCH)).budget == 32768
        assert (await budget.usage("m", BRANCH)).budget == 32768
        assert lookups == ["m"]

    async def test_looks_the_context_length_up_per_model(self) -> None:
        _, budget, lookups = attached(ContextSettings(), context_length=32768)

        await budget.usage("m", BRANCH)
        await budget.usage("other", BRANCH)

        assert lookups == ["m", "other"]

    async def test_looks_the_context_length_up_again_while_the_model_cannot_tell(self) -> None:
        lengths = iter([None, 32768, 65536])
        budget = ContextBudget(ContextSettings(), lambda _: next(lengths))

        budgets = [(await budget.usage("m", BRANCH)).budget for _ in range(3)]

        assert budgets == [None, 32768, 32768]

    async def test_looks_the_context_length_up_off_the_event_loop(self) -> None:
        threads: list[threading.Thread] = []

        def lookup(_: str) -> int:
            threads.append(threading.current_thread())
            return 32768

        await ContextBudget(ContextSettings(), lookup).usage("m", BRANCH)

        assert threads != [threading.current_thread()]

    async def test_a_conversation_not_started_takes_nothing(self) -> None:
        budget = ContextBudget(ContextSettings(budget=8192), lambda _: None)

        assert await budget.usage("m", None) == ContextUsage(0, 8192, 0.8)

    async def test_a_branch_not_measured_yet_has_an_unknown_usage(self) -> None:
        budget = ContextBudget(ContextSettings(budget=8192), lambda _: None)

        assert await budget.usage("m", BRANCH) == ContextUsage(None, 8192, 0.8)

    async def test_keeps_the_usage_of_each_branch(self) -> None:
        bus, budget, _ = attached(ContextSettings(budget=8192))

        bus.emit(chat_call(1000, branch_id=BRANCH))
        bus.emit(chat_call(3000, branch_id=OTHER_BRANCH))

        assert (await budget.usage("m", BRANCH)).used == 1000
        assert (await budget.usage("m", OTHER_BRANCH)).used == 3000

    async def test_forgets_a_removed_branch(self) -> None:
        bus, budget, _ = attached(ContextSettings(budget=8192))
        bus.emit(chat_call(1000, branch_id=BRANCH))
        bus.emit(chat_call(3000, branch_id=OTHER_BRANCH))

        budget.forget(BRANCH)
        budget.forget(uuid4())

        assert (await budget.usage("m", BRANCH)).used is None
        assert (await budget.usage("m", OTHER_BRANCH)).used == 3000
