from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sensai.monitoring.interaction import Interaction

if TYPE_CHECKING:
    from collections.abc import Callable

    from sensai.config.settings import ContextSettings
    from sensai.core.events import EventBus


@dataclass(frozen=True)
class ContextUsage:
    """Tokens the conversation takes in the context of the model, against the budget it may take."""

    used: int
    budget: int | None
    """None when neither the settings nor the model give one."""
    warning_threshold: float

    @property
    def ratio(self) -> float | None:
        """Share of the budget in use, None without a budget."""
        return None if self.budget is None else self.used / self.budget

    @property
    def near_limit(self) -> bool:
        """Whether the usage has reached the warning threshold of the budget."""
        return self.ratio is not None and self.ratio >= self.warning_threshold


class ContextBudget:
    """Observer of the `EventBus` measuring the context the conversation takes after every call to the model.

    The measure is the real token count the provider reports for the last call of a chat: its prompt, holding the
    whole conversation, plus its answer. The calls made outside of a conversation branch, such as `/ask`, do not
    count.
    """

    _settings: ContextSettings
    _context_length: Callable[[str], int | None]
    _on_change: Callable[[ContextUsage], object]
    _lengths: dict[str, int | None]
    _used: int

    def __init__(
        self,
        settings: ContextSettings,
        context_length: Callable[[str], int | None],
        on_change: Callable[[ContextUsage], object],
    ) -> None:
        """Measure against the budget of `settings`, else the `context_length` of the model, reporting to `on_change`.

        `context_length` is looked up once per model.
        """
        self._settings = settings
        self._context_length = context_length
        self._on_change = on_change
        self._lengths = {}
        self._used = 0

    def attach(self, bus: EventBus) -> None:
        """Measure every interaction emitted on `bus` from now on."""
        bus.subscribe(Interaction, self.record)

    def record(self, interaction: Interaction) -> None:
        """Take the token count of `interaction` as the new usage, when it is a chat call that reports one."""
        prompt_tokens = interaction.metrics.prompt_tokens
        if interaction.branch_id is None or prompt_tokens is None:
            return
        self._used = prompt_tokens + (interaction.metrics.completion_tokens or 0)
        self._on_change(self.usage(interaction.model))

    def usage(self, model: str) -> ContextUsage:
        """Return the last measured usage against the budget of `model`."""
        return ContextUsage(self._used, self._budget(model), self._settings.warning_threshold)

    def _budget(self, model: str) -> int | None:
        if self._settings.budget is not None:
            return self._settings.budget
        if model not in self._lengths:
            self._lengths[model] = self._context_length(model)
        return self._lengths[model]
