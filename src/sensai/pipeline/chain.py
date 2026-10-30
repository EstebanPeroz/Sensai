from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sensai.llm.responses import ChatResponse
from sensai.monitoring.interaction import ignore_interaction

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from uuid import UUID

    from sensai.monitoring.interaction import Interaction
    from sensai.ui.adapter import UIAdapter


@dataclass
class ChatContext:
    """Everything an exchange carries on its way down to the model."""

    message: str
    ui: UIAdapter
    messages: list[dict] = field(default_factory=list)
    branch_id: UUID | None = None
    """Branch of the conversation the messages come from, oldest first."""
    on_interaction: Callable[[Interaction], None] = ignore_interaction
    """Called once for every call to the model this exchange makes, whatever its outcome."""


type Next = Callable[[ChatContext], Iterator[ChatResponse]]
"""The rest of the chain, ending on the model: call it to go on, drop it to answer instead."""

type Stage = Callable[[ChatContext, Next], Iterator[ChatResponse]]
"""One step of the chain, wrapped around every step that follows it."""


def answer_chunk(content: str) -> ChatResponse:
    """Build a final answer chunk, for a stage that replies without reaching the model."""
    return ChatResponse({"done": True, "message": {"role": "assistant", "content": content}})
