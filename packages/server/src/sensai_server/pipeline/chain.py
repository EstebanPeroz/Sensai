"""Core types of the chat pipeline.

Stages are async generators, so moving them from the legacy sync code has a few pitfalls. Every stage follows
these rules:

1. No `yield from` in an async generator. Relay the rest of the chain with:

       async with aclosing(next_stage(context)) as chunks:
           async for chunk in chunks:
               yield chunk

2. Always wrap `next_stage(...)` in `contextlib.aclosing`. Closing the outer generator does not close the inner
   one, it is only finalised later by the GC. `aclosing` makes the closing cascade deterministic down to the
   model, which must close its HTTP stream.
3. Never swallow `asyncio.CancelledError` or `GeneratorExit`. Clean up in a `finally`, but let them propagate.
4. Blocking work (DB, HTTP, embeddings) goes in a thread, so the event loop shared by every session never blocks.
5. A stage that answers in place of the model yields its chunks, the last with `done=True`, then returns without
   calling `next_stage`.
6. A stage only changes `ChatContext.messages`. `model`, `persona` and `out` are read-only by convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sensai_server.chat.react import EFFORTS
from sensai_server.llm.responses import ChatResponse
from sensai_server.monitoring.interaction import ignore_interaction

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from uuid import UUID

    from sensai_server.app.outbound import Outbound
    from sensai_server.cache import SessionCache
    from sensai_server.llm.persona import Persona
    from sensai_server.monitoring.interaction import Interaction
    from sensai_server.tools.state import ToolState


@dataclass
class ChatContext:
    """Everything an exchange carries on its way down to the model.

    The pipeline is shared by every session, so all that is specific to one exchange travels here.
    """

    message: str
    out: Outbound
    model: str
    persona: Persona | None = None
    messages: list[dict] = field(default_factory=list)
    tools: ToolState | None = None
    max_iterations: int = EFFORTS["off"]
    rag_id: UUID | None = None
    cache: SessionCache | None = None
    on_interaction: Callable[[Interaction], None] = ignore_interaction
    """Called once for every call to the model this exchange makes, whatever its outcome. Sync, never awaited."""


type Next = Callable[[ChatContext], AsyncGenerator[ChatResponse]]
"""The rest of the chain, ending on the model: call it to go on, drop it to answer instead."""

type Stage = Callable[[ChatContext, Next], AsyncGenerator[ChatResponse]]
"""One step of the chain, wrapped around every step that follows it."""


def answer_chunk(content: str) -> ChatResponse:
    """Build a final answer chunk, for a stage that replies without reaching the model."""
    return ChatResponse({"done": True, "message": {"role": "assistant", "content": content}})
