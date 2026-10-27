from __future__ import annotations

import re
from typing import TYPE_CHECKING

from sensai.pipeline.chain import answer_chunk

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sensai.llm.responses import ChatResponse
    from sensai.pipeline.chain import ChatContext, Next

WORD = re.compile(r"[\w']+")

BAD_WORDS = frozenset({"fuck", "fucking", "bitch", "cunt", "asshole"})
WARNING = "Let's keep this civil. Tell me what you need and I'll help."


class BadWordsGuard:
    """Stop an abusive message at the door, with a short reply instead of an answer."""

    def __init__(self, bad_words: frozenset[str] = BAD_WORDS) -> None:
        """Keep the words this guard watches for, so settings can widen or narrow the list."""
        self._bad_words = bad_words

    def __call__(self, context: ChatContext, next_stage: Next) -> Iterator[ChatResponse]:
        """Answer the warning and go no further on an abusive message, or let the chain go on."""
        if self._is_abusive(context.message):
            yield answer_chunk(WARNING)
            return
        yield from next_stage(context)

    def _is_abusive(self, message: str) -> bool:
        """Report whether any word of `message` is one this guard watches for."""
        return bool(self._bad_words & {word.lower() for word in WORD.findall(message)})
