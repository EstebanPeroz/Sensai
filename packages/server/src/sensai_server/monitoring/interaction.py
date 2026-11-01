import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sensai_server.llm.responses import ChatMetrics

if TYPE_CHECKING:
    from sensai_server.events import EventBus

logger = logging.getLogger(__name__)


class InteractionStatus(StrEnum):
    """How a call to the model ended."""

    OK = "ok"
    ERROR = "error"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class Interaction:
    """Structured record of one call to the model: what was sent, what came back, and how long it took.

    It is also the event emitted on the `EventBus` once the call is over, whatever its outcome.
    `latency_ms` is the response time, from the request to the last chunk; `first_token_ms` is the
    wait for the first one, the latency the user feels.
    """

    model: str
    status: InteractionStatus
    latency_ms: float
    branch_id: UUID | None = None
    persona: str | None = None
    request: list[dict] | None = None
    response: str | None = None
    thinking: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    error: str | None = None
    first_token_ms: float | None = None
    metrics: ChatMetrics = field(default_factory=ChatMetrics)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


def ignore_interaction(_: Interaction) -> None:
    """Ignore a call to the model, when nobody records them."""


def emit_interaction(events: EventBus, interaction: Interaction) -> None:
    """Emit `interaction` on `events`, never raising.

    It runs in the `finally` of a model call that may be closing, so a failing subscriber is only logged, and the
    caller goes on.
    """
    try:
        events.emit(interaction)
    except Exception:
        logger.exception("Failed to emit an interaction")
