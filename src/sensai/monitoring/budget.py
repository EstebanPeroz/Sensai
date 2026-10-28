from __future__ import annotations

from dataclasses import dataclass


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
