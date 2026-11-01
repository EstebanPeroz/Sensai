from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING

from sensai_server.cache import CacheError
from sensai_server.llm.error import EmbeddingError
from sensai_server.pipeline.chain import answer_chunk

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sensai_server.cache import SessionCache
    from sensai_server.llm.responses import ChatResponse
    from sensai_server.pipeline.chain import ChatContext, Next


class CacheStage:
    """Answer from the session's semantic cache when it holds a prompt meaning the same, and fill it on a miss.

    The cache comes from the context, so this stage holds no state per session and one instance is shared by all.
    """

    async def __call__(self, context: ChatContext, next_stage: Next) -> AsyncGenerator[ChatResponse]:
        """Stream the cached answer to the message, or the rest of the chain's, storing the latter once complete.

        Only a complete answer is stored: one that ended on a `done` chunk without error and is not blank. A cache
        that cannot be read or written is reported, never raised: it must never prevent an answer.
        """
        cache = context.cache
        if cache is None:
            async with aclosing(next_stage(context)) as chunks:
                async for chunk in chunks:
                    yield chunk
            return
        cached = await self._lookup(context, cache)
        if cached is not None:
            yield answer_chunk(cached)
            return
        content = ""
        complete = False
        async with aclosing(next_stage(context)) as chunks:
            async for chunk in chunks:
                if chunk.content is not None:
                    content += chunk.content
                complete = chunk.done and chunk.error is None
                yield chunk
        if complete and content.strip():
            await self._store(context, cache, content)

    @staticmethod
    async def _lookup(context: ChatContext, cache: SessionCache) -> str | None:
        """Return the cached answer to the message, reporting a failed lookup as a miss rather than raising."""
        try:
            return await cache.lookup(context.message)
        except (CacheError, EmbeddingError) as error:
            await context.out.send_system_message(f"The semantic cache could not be read: {error.message}")
            return None

    @staticmethod
    async def _store(context: ChatContext, cache: SessionCache, answer: str) -> None:
        """Cache `answer` to the message, reporting a failed store rather than raising."""
        try:
            await cache.store(context.message, answer)
        except (CacheError, EmbeddingError) as error:
            await context.out.send_system_message(f"The semantic cache could not be written: {error.message}")
