from __future__ import annotations

import contextlib
import hashlib
import math
import uuid
from array import array
from typing import TYPE_CHECKING

import redis

from sensai.error import SensaiError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from redis.typing import EncodableT, FieldT

    from sensai.config.settings import CacheSettings
    from sensai.memory.semantic_cache_repository import CacheScope

KEY_PREFIX = "sensai:cache:"
_FIELDS = ("embedding", "context_embedding", "answer")


class RedisSemanticCache:
    """Semantic cache kept in Redis: one hash per answer, holding the query, its embedding and the answer.

    Entries expire after the configured TTL. Lookups compare the query embedding with every entry of the scope;
    a Redis or embedding failure is treated as a miss, so the chat keeps working without the cache. Keys also carry
    the embedding model, since vectors from different models are not comparable.
    """

    def __init__(
        self,
        client: redis.Redis,
        settings: CacheSettings,
        embed: Callable[[str, str], Sequence[float]],
        embedding_model: str,
    ) -> None:
        """Init the cache with its Redis client, its settings, the embedding function and the model it embeds with."""
        self._client = client
        self._threshold = settings.similarity_threshold
        self._ttl = settings.ttl
        self._embed = embed
        self._embedding_model = embedding_model

    def lookup(self, query: str, scope: CacheScope, context: str = "") -> str | None:
        """Return the cached answer of the most similar query of the scope, or None below the similarity threshold.

        The query and its context are compared separately, and both must reach the threshold: comparing them as one
        text would let a long shared context hide a different query. A query without context only matches entries
        without context, and the other way around.
        """
        embedding = self._embed_query(query)
        context_embedding = self._embed_query(context) if context else []
        if not embedding or (context and not context_embedding):
            return None
        best_answer: str | None = None
        best_score = self._threshold
        try:
            entries = self._scope_entries(scope)
        except redis.RedisError:
            return None
        for stored, stored_context, answer in entries:
            if not isinstance(stored, bytes) or not isinstance(answer, bytes):
                continue
            score = _cosine_similarity(embedding, _unpack(stored))
            if context_embedding or stored_context:
                if not context_embedding or not isinstance(stored_context, bytes) or not stored_context:
                    continue
                score = min(score, _cosine_similarity(context_embedding, _unpack(stored_context)))
            if score >= best_score:
                best_answer, best_score = answer.decode(), score
        return best_answer

    def store(self, query: str, scope: CacheScope, answer: str, context: str = "") -> None:
        """Cache the answer to a query asked after the given context, within its scope, for the configured TTL."""
        embedding = self._embed_query(query)
        context_embedding = self._embed_query(context) if context else []
        if not embedding or (context and not context_embedding):
            return
        key = self._scope_prefix(scope) + uuid.uuid4().hex
        entry: dict[FieldT, EncodableT] = {
            "query": query,
            "embedding": _pack(embedding),
            "context": context,
            "context_embedding": _pack(context_embedding),
            "answer": answer,
        }
        with contextlib.suppress(redis.RedisError):
            pipeline = self._client.pipeline()
            pipeline.hset(key, mapping=entry)
            pipeline.expire(key, self._ttl)
            pipeline.execute()

    def clear(self) -> None:
        """Remove every cached answer, whatever its scope or embedding model."""
        with contextlib.suppress(redis.RedisError):
            keys = list(self._client.scan_iter(match=KEY_PREFIX + "*"))
            if keys:
                self._client.delete(*keys)

    def _scope_entries(self, scope: CacheScope) -> list[list[bytes | None]]:
        """Return the fields compared on lookup of every entry of the scope, read in a single round trip."""
        pipeline = self._client.pipeline(transaction=False)
        for key in self._client.scan_iter(match=self._scope_prefix(scope) + "*"):
            pipeline.hmget(key, _FIELDS)
        return pipeline.execute()

    def _embed_query(self, query: str) -> Sequence[float]:
        try:
            return self._embed(query, self._embedding_model)
        except SensaiError:
            return []

    def _scope_prefix(self, scope: CacheScope) -> str:
        fields = (self._embedding_model, scope.model, scope.persona, scope.subject or "")
        digest = hashlib.sha256("\0".join(fields).encode()).hexdigest()[:16]
        return f"{KEY_PREFIX}{digest}:"


def _pack(vector: Sequence[float]) -> bytes:
    return array("f", vector).tobytes()


def _unpack(data: bytes) -> list[float]:
    return array("f", data).tolist()


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    norms = math.hypot(*a) * math.hypot(*b)
    if norms == 0:
        return 0.0
    return math.sumprod(a, b) / norms
