"""The semantic cache of every session, on Redis: one shared `SemanticCache`, one `SessionCache` handle per session.

Each entry is a HASH under `<namespace>:<client_id>:<entry uuid>`, holding the `session` it belongs to, the `prompt`,
the `answer` and the prompt's `embedding` (FLOAT32 bytes), and expiring after the TTL. One search index,
`<namespace>:idx`, covers every entry: a KNN search pre-filtered on the `session` tag finds the closest prompt of one
session only.

Redis calls go through the async client, and embedding through `asyncio.to_thread`, so the event loop never blocks.
"""

from __future__ import annotations

import asyncio
import struct
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from redis.asyncio import Redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import RedisError, ResponseError

from sensai_server.cache.error import CacheError
from sensai_server.cache.session import SessionCache

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from uuid import UUID

    from redis.commands.search.result import Result
    from redis.typing import EncodableT, FieldT

    from sensai_server.llm.embedder import Embedder

NAMESPACE = "sensai:cache"
_INDEX_NOT_FOUND = "SEARCH_INDEX_NOT_FOUND"
_SCAN_COUNT = 500


def _to_bytes(vector: list[float]) -> bytes:
    """Pack `vector` the way a FLOAT32 vector field stores it: little-endian 32-bit floats."""
    return struct.pack(f"<{len(vector)}f", *vector)


def _session_tag(client_id: UUID) -> str:
    """Return the `session` tag of `client_id`: its hex form, which has no character a tag query must escape."""
    return client_id.hex


class SemanticCache:
    """The Redis store shared by every session, handing each one its own `SessionCache`.

    The search index is created on the first `store`, once the size of the embedder's vectors is known. When that
    size changes (the embedding model did), the cache is disposable: the index and every entry are dropped, then
    the index is created again.
    """

    _client: Redis
    _embedder: Embedder
    _threshold: float
    _ttl: int
    _namespace: str
    _dimension: int | None

    def __init__(
        self,
        redis_url: str,
        embedder: Embedder,
        similarity_threshold: float,
        ttl: int,
        *,
        namespace: str = NAMESPACE,
    ) -> None:
        """Keep the client of the Redis at `redis_url`, without connecting yet, and how entries are matched and kept.

        A prompt hits when its cosine similarity to a cached one is at least `similarity_threshold`, and an entry
        expires `ttl` seconds after it is stored. Every key is under `namespace`.
        """
        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._embedder = embedder
        self._threshold = similarity_threshold
        self._ttl = ttl
        self._namespace = namespace
        self._dimension = None

    @property
    def similarity_threshold(self) -> float:
        """The cosine similarity from which a cached prompt counts as the same."""
        return self._threshold

    async def connect(self) -> None:
        """Check that Redis answers, before the cache is used.

        Raises:
            CacheError: If Redis cannot be reached.

        """
        if not await self.ping():
            msg = "the Redis server does not answer"
            raise CacheError(msg)

    async def ping(self) -> bool:
        """Report whether Redis answers."""
        try:
            return bool(await self._client.ping())
        except RedisError:
            return False

    def for_session(self, client_id: UUID) -> SessionCache:
        """Hand the session `client_id` its view of the cache. Cheap, with no I/O."""
        return SessionCache(self, client_id)

    async def clear_all(self) -> None:
        """Delete every entry of every session.

        Raises:
            CacheError: If Redis fails.

        """
        await self._unlink_matching(f"{self._namespace}:*")

    async def close(self) -> None:
        """Close the Redis client."""
        await self._client.aclose()

    async def embed(self, prompt: str) -> list[float]:
        """Embed `prompt` in a thread, raising `EmbeddingError` when the embedder fails."""
        [vector] = await asyncio.to_thread(self._embedder.embed, [prompt])
        return vector

    async def closest(self, client_id: UUID, vector: list[float]) -> tuple[str, float] | None:
        """Return the answer to the prompt of `client_id` closest to `vector`, and its similarity, if there is one.

        No index yet, or an index of another vector size, means nothing to find.

        Raises:
            CacheError: If Redis fails.

        """
        if await self._index_dimension() != len(vector):
            return None
        query = (
            Query(f"@session:{{{_session_tag(client_id)}}}=>[KNN 1 @embedding $vec AS distance]")
            .sort_by("distance")
            .return_fields("answer", "distance")
            .paging(0, 1)
            .dialect(2)
        )
        try:
            search = self._client.ft(self._index).search(query, query_params={"vec": _to_bytes(vector)})
            result = cast("Result", await self._run(search))
        except CacheError:
            self._dimension = None
            raise
        if not result.docs:
            return None
        [doc] = result.docs
        return str(doc.answer), 1 - float(doc.distance)

    async def store_entry(self, client_id: UUID, prompt: str, answer: str, vector: list[float]) -> None:
        """Write one entry of `client_id`, expiring after the TTL.

        Raises:
            CacheError: If Redis fails.

        """
        await self._ensure_index(len(vector))
        key = f"{self._namespace}:{client_id}:{uuid4()}"
        mapping: dict[FieldT, EncodableT] = {
            "session": _session_tag(client_id),
            "prompt": prompt,
            "answer": answer,
            "embedding": _to_bytes(vector),
        }
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.hset(key, mapping=mapping)
            pipe.expire(key, self._ttl)
            await self._run(pipe.execute())

    async def clear_session(self, client_id: UUID) -> None:
        """Delete every entry of `client_id`.

        Raises:
            CacheError: If Redis fails.

        """
        await self._unlink_matching(f"{self._namespace}:{client_id}:*")

    @property
    def _index(self) -> str:
        return f"{self._namespace}:idx"

    async def _index_dimension(self) -> int | None:
        """Return the vector size of the search index, or None when there is no index yet.

        Raises:
            CacheError: If Redis fails.

        """
        if self._dimension is None:
            try:
                info = await self._client.ft(self._index).info()
            except ResponseError as error:
                if _INDEX_NOT_FOUND in str(error):
                    return None
                raise CacheError(str(error)) from error
            except RedisError as error:
                raise CacheError(str(error)) from error
            self._dimension = _vector_dimension(cast("dict", info))
        return self._dimension

    async def _ensure_index(self, dimension: int) -> None:
        """Create the search index for vectors of `dimension`, dropping one of another size and its entries first.

        Raises:
            CacheError: If Redis fails.

        """
        current = await self._index_dimension()
        if current == dimension:
            return
        if current is not None:
            await self._run(self._client.ft(self._index).dropindex(delete_documents=True))
            self._dimension = None
        fields = [
            TagField("session"),
            VectorField("embedding", "FLAT", {"TYPE": "FLOAT32", "DIM": dimension, "DISTANCE_METRIC": "COSINE"}),
        ]
        definition = IndexDefinition(prefix=[f"{self._namespace}:"], index_type=IndexType.HASH)
        await self._run(self._client.ft(self._index).create_index(fields, definition=definition))
        self._dimension = dimension

    async def _unlink_matching(self, pattern: str) -> None:
        """Unlink every key matching `pattern`, a batch per scanned page.

        Raises:
            CacheError: If Redis fails.

        """
        try:
            batch: list[str] = []
            async for key in self._client.scan_iter(match=pattern, count=_SCAN_COUNT):
                batch.append(key)
                if len(batch) >= _SCAN_COUNT:
                    await self._client.unlink(*batch)
                    batch = []
            if batch:
                await self._client.unlink(*batch)
        except RedisError as error:
            raise CacheError(str(error)) from error

    @staticmethod
    async def _run[T](call: Awaitable[T]) -> T:
        """Await the Redis `call`, wrapping a Redis failure in `CacheError`."""
        try:
            return await call
        except RedisError as error:
            raise CacheError(str(error)) from error


def _vector_dimension(info: dict) -> int | None:
    """Read the size of the `embedding` vector field out of `FT.INFO`'s answer."""
    for attribute in info.get("attributes", []):
        fields = dict(zip(attribute[::2], attribute[1::2], strict=False))
        if fields.get("identifier") == "embedding":
            return int(fields["dim"])
    return None
