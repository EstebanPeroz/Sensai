from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sensai.llm.responses import ChatMetrics


class InteractionStatus(StrEnum):
    """How a call to the model ended."""

    OK = "ok"
    ERROR = "error"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class Interaction:
    """Structured record of one call to the model: what was sent, what came back, and how long it took.

    It is also the event emitted on the `EventBus` once the call is over, whatever its outcome.
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
