from __future__ import annotations

from contextlib import aclosing
from functools import partial
from typing import TYPE_CHECKING

from sensai_protocol.error import SensaiError

from sensai_server.llm.threaded import iterate_in_thread

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sensai_server.llm.provider import ProviderManager
    from sensai_server.llm.responses import ChatResponse
    from sensai_server.pipeline.chain import ChatContext
    from sensai_server.tools.tool_manager import ToolManager


class ModelStage:
    """Send the payload the earlier stages built to the provider serving the model, and stream what it answers."""

    _tool_manager: ToolManager
    _provider_manager: ProviderManager

    def __init__(self, provider_manager: ProviderManager, tool_manager: ToolManager) -> None:
        """Keep the manager this stage finds the provider of each exchange's model in, and the tools ReAct will call."""
        self._provider_manager = provider_manager
        self._tool_manager = tool_manager

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
        # It will build `think` from `session.effort`, put `self._tool_manager.get_tool_call_rules(context.tools)` in
        # the payload as `tools`, and run the model's tool calls with `self._tool_manager.calls`.
        payload = {"model": context.model, "messages": list(context.messages)}
        async with aclosing(iterate_in_thread(partial(provider.chat, payload, stream=True))) as chunks:
            async for chunk in chunks:
                yield chunk
