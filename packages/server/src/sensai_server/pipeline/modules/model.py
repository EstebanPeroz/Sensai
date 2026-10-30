from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sensai_server.chat.react import ReAct
    from sensai_server.llm.responses import ChatResponse
    from sensai_server.pipeline.chain import ChatContext


class ModelStage:
    """Hand the payload the earlier stages built to the ReAct loop, and stream what it answers."""

    _react: ReAct

    def __init__(self, react: ReAct) -> None:
        """Keep the ReAct loop, shared by every session, this stage hands each exchange to."""
        self._react = react

    async def __call__(self, context: ChatContext) -> AsyncGenerator[ChatResponse]:
        """Stream the answer to the payload in `context`.

        Takes no `next_stage`: nothing runs after the model, it closes the chain. Raises `SensaiError` when no
        provider serves the model, and lets the provider's errors propagate.
        """
        async with aclosing(self._react.generate_answer(context)) as chunks:
            async for chunk in chunks:
                yield chunk
