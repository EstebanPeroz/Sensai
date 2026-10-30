from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sensai_server.llm.responses import ChatResponse
    from sensai_server.pipeline.chain import ChatContext, Next


class PersonaStage:
    """Prepend the conversation's persona to the payload, then let the chain go on.

    The persona comes from the context, a snapshot of the conversation's own: it is never written to the history.
    """

    async def __call__(self, context: ChatContext, next_stage: Next) -> AsyncGenerator[ChatResponse]:
        """Add the persona to `context` when there is one, and stream what the rest of the chain answers."""
        if context.persona is not None:
            context.messages = [context.persona.to_message().to_dict(), *context.messages]
        async with aclosing(next_stage(context)) as chunks:
            async for chunk in chunks:
                yield chunk
