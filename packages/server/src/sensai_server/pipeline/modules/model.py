from __future__ import annotations

from contextlib import aclosing
from functools import partial
from typing import TYPE_CHECKING

from sensai_protocol.error import SensaiError

from sensai_server.pipeline.modules._threaded import iterate_in_thread

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sensai_server.llm.provider import ProviderManager
    from sensai_server.llm.responses import ChatResponse
    from sensai_server.pipeline.chain import ChatContext


class ModelStage:
    """Send the payload the earlier stages built to the provider serving the model, and stream what it answers."""

    def __init__(self, provider_manager: ProviderManager) -> None:
        """Keep the manager this stage finds the provider of each exchange's model in."""
        self._provider_manager = provider_manager

    async def __call__(self, context: ChatContext) -> AsyncGenerator[ChatResponse]:
        """Stream the answer to the payload in `context`.

        Takes no `next_stage`: nothing runs after the model, it closes the chain. Raises `SensaiError` when no
        provider serves the model, and lets the provider's errors propagate.
        """
        provider = self._provider_manager.get_provider(context.model)
        if provider is None:
            msg = f"No provider serves model {context.model}"
            raise SensaiError(msg)

        # TODO(react): ReAct replaces this direct call  # noqa: TD003, FIX002
        # It will build `think` and `tools` from `session.effort`.
        payload = {"model": context.model, "messages": list(context.messages)}
        async with aclosing(iterate_in_thread(partial(provider.chat, payload, stream=True))) as chunks:
            async for chunk in chunks:
                yield chunk
