from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sensai_server.monitoring.budget import ContextBudget
    from sensai_server.session import Session


async def send_context_usage(session: Session, budget: ContextBudget | None) -> None:
    """Send the session the context usage of its current branch, against the budget of its model.

    Nothing is sent without a `budget` or without a model.
    """
    conv = session.conv
    if budget is None or conv.model is None:
        return
    await session.out.update_context_usage(await budget.usage(conv.model, conv.uuid))
