from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from sensai.llm.message import Role
from sensai.llm.responses import ChatResponse
from sensai.semantic.cache_repository import CacheScope

if TYPE_CHECKING:
    from sensai.llm.persona import Persona
    from sensai.memory.history import History
    from sensai.semantic.cache_repository import SemanticCacheRepository

# Persona a cached answer is scoped to when none is active.
NO_PERSONA = ""
# Previous user messages a query is cached with: the same question can call for another answer in another context.
CACHE_CONTEXT_TURNS = 2


class CacheKey(NamedTuple):
    """What a query is looked up and cached with: the query itself, its scope and the previous user messages."""

    query: str
    scope: CacheScope
    context: str


def cache_key(query: str, model: str, persona: Persona | None, history: History) -> CacheKey:
    """Return what `query`, asked to `model` with `persona` after `history`, is looked up and cached with.

    The scope is read on every query, as /model and /persona change the model and persona mid-conversation.
    """
    previous = [message.content for message in history.messages() if message.role is Role.USER]
    scope = CacheScope(model=model, persona=persona.name if persona is not None else NO_PERSONA)
    return CacheKey(query, scope, "\n".join(previous[-CACHE_CONTEXT_TURNS:]))


def cached_response(cache: SemanticCacheRepository, key: CacheKey) -> ChatResponse | None:
    """Return the cached answer to the query of `key` as a complete model response, or None if there is none."""
    answer = cache.lookup(key.query, key.scope, key.context)
    if answer is None:
        return None
    message = {"role": Role.ASSISTANT.value, "content": answer}
    return ChatResponse({"model": key.scope.model, "done": True, "message": message})
