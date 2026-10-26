from __future__ import annotations

import contextlib
import hashlib
import math
import uuid
from array import array
from typing import TYPE_CHECKING

import redis

from sensai.error import SensaiError
from sensai.semantic.cache_repository import SemanticCacheRepository

if TYPE_CHECKING:
    from collections.abc import Sequence

    from redis.typing import EncodableT, FieldT

    from sensai.config.settings import CacheSettings
    from sensai.llm.embedder import Embedder
    from sensai.semantic.cache_repository import CacheScope

KEY_PREFIX = "sensai:cache:"
SCAN_COUNT = 1000
_FIELDS = ("embedding", "context_embedding", "answer")
# Lets texts holding lone surrogates, such as undecodable terminal input, be cached and read back unchanged.
_ENCODING_ERRORS = "surrogatepass"


class RedisSemanticCache(SemanticCacheRepository):
    """Semantic cache kept in Redis: one hash per answer, holding the query, its embedding and the answer.

    Entries expire after the configured TTL. Lookups compare the query embedding with every entry of the scope;
    a Redis or embedding failure, or an answer that cannot be decoded, is treated as a miss, so the chat keeps working
    without the cache. Keys also carry the embedding model, since vectors from different models are not comparable.
    """

    def __init__(self, client: redis.Redis, settings: CacheSettings, embedder: Embedder) -> None:
        """Init the cache with its Redis client, its settings and the embedder its queries are compared with."""
        self._client = client
        self._threshold = settings.similarity_threshold
        self._ttl = settings.ttl
        self._embedder = embedder

    def lookup(self, query: str, scope: CacheScope, context: str = "") -> str | None:
        """Return the cached answer of the most similar query of the scope, or None below the similarity threshold.

        The query and its context are compared separately, and both must reach the threshold: comparing them as one
        text would let a long shared context hide a different query. An entry cached without context, a question asked
        first, stands on its own and matches whatever the context; an entry cached with context only matches a query
        asked after a similar one.
        """
        embeddings = self._embed(query, context)
        if embeddings is None:
            return None
        embedding, context_embedding = embeddings
        best_answer: str | None = None
        best_score = self._threshold
        try:
            entries = self._scope_entries(scope)
        except redis.RedisError:
            return None
        for stored, stored_context, answer in entries:
            if not isinstance(stored, bytes) or not isinstance(stored_context, bytes) or not isinstance(answer, bytes):
                continue
            score = self._cosine_similarity(embedding, self._unpack(stored))
            if stored_context:
                if not context_embedding:
                    continue
                score = min(score, self._cosine_similarity(context_embedding, self._unpack(stored_context)))
            if score >= best_score:
                with contextlib.suppress(UnicodeDecodeError):
                    best_answer, best_score = answer.decode(errors=_ENCODING_ERRORS), score
        return best_answer

    def store(self, query: str, scope: CacheScope, answer: str, context: str = "") -> None:
        """Cache the answer to a query asked after the given context, within its scope, for the configured TTL."""
        embeddings = self._embed(query, context)
        if embeddings is None:
            return
        embedding, context_embedding = embeddings
        key = self._scope_prefix(scope) + uuid.uuid4().hex
        entry: dict[FieldT, EncodableT] = {
            "query": self._encode(query),
            "embedding": self._pack(embedding),
            "context": self._encode(context),
            "context_embedding": self._pack(context_embedding),
            "answer": self._encode(answer),
        }
        with contextlib.suppress(redis.RedisError):
            pipeline = self._client.pipeline()
            pipeline.hset(key, mapping=entry)
            pipeline.expire(key, self._ttl)
            pipeline.execute()

    def _scope_entries(self, scope: CacheScope) -> list[list[bytes | None]]:
        """Return the fields compared on lookup of every entry of the scope, read in a single round trip."""
        pipeline = self._client.pipeline(transaction=False)
        for key in self._client.scan_iter(match=self._scope_prefix(scope) + "*", count=SCAN_COUNT):
            pipeline.hmget(key, _FIELDS)
        return pipeline.execute()

    def _embed(self, query: str, context: str) -> tuple[list[float], list[float]] | None:
        """Embed the query and its context in a single call, or return None if the embedder fails."""
        try:
            vectors = self._embedder.embed([query, context] if context else [query])
        except SensaiError:
            return None
        return vectors[0], vectors[1] if context else []

    def _scope_prefix(self, scope: CacheScope) -> str:
        fields = (self._embedder.model, scope.model, scope.persona)
        digest = hashlib.sha256("\0".join(fields).encode()).hexdigest()[:16]
        return f"{KEY_PREFIX}{digest}:"

    @staticmethod
    def _encode(text: str) -> bytes:
        return text.encode(errors=_ENCODING_ERRORS)

    @staticmethod
    def _pack(vector: Sequence[float]) -> bytes:
        return array("f", vector).tobytes()

    @staticmethod
    def _unpack(data: bytes) -> list[float]:
        """Decode a stored vector, or return an empty one, similar to nothing, if Redis holds malformed bytes."""
        try:
            return array("f", data).tolist()
        except ValueError:
            return []

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b):
            return 0.0
        norms = math.hypot(*a) * math.hypot(*b)
        if norms == 0:
            return 0.0
        return math.sumprod(a, b) / norms
