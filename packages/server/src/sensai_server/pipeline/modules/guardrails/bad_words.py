from __future__ import annotations

import re
from contextlib import aclosing
from typing import TYPE_CHECKING

from sensai_server.pipeline.chain import answer_chunk

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sensai_server.llm.responses import ChatResponse
    from sensai_server.pipeline.chain import ChatContext, Next

WORD = re.compile(r"[\w']+")

BAD_WORDS = frozenset({"fuck", "fucking", "bitch", "cunt", "asshole"})
WARNING = "Let's keep this civil. Tell me what you need and I'll help."


class BadWordsGuard:
    """Stop an abusive message at the door, with a short reply instead of an answer."""

    def __init__(self, bad_words: frozenset[str] = BAD_WORDS) -> None:
        """Keep the words this guard watches for, so settings can widen or narrow the list."""
        self._bad_words = bad_words

    async def __call__(self, context: ChatContext, next_stage: Next) -> AsyncGenerator[ChatResponse]:
        """Answer the warning and go no further on an abusive message, or let the chain go on."""
        if self._is_abusive(context.message):
            yield answer_chunk(WARNING)
            return
        async with aclosing(next_stage(context)) as chunks:
            async for chunk in chunks:
                yield chunk

    def _is_abusive(self, message: str) -> bool:
        """Report whether any word of `message` is one this guard watches for."""
        return bool(self._bad_words & {word.lower() for word in WORD.findall(message)})
