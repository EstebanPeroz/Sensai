"""The RAG of every conversation: index and search by RAG id, and keep each branch linked to its RAG id.

One `Rag` is shared by every session. Each RAG id names its own Qdrant collection, and a branch is linked to one
RAG id, stored in the database and cached here.

Every Qdrant, embedding and database call goes through `asyncio.to_thread`. An embedded Qdrant (`path=` or
":memory:") is not documented as thread-safe, so the Qdrant calls are serialised by a lock. Embedding stays outside
of it, so that two sessions can embed at the same time.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sensai_server.rag.chunking import Chunker
from sensai_server.rag.collection import Collection, as_chunk, by_source, chunk_point
from sensai_server.rag.error import RagError

if TYPE_CHECKING:
    from collections.abc import Callable

    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    from sensai_server.llm.embedder import Embedder
    from sensai_server.memory.persistent_db import PersistentDatabase

SEARCH_LIMIT = 5
MIN_SCORE = 0.5


class Rag:
    """Index and search the documents of every conversation, one Qdrant collection per RAG id."""

    _client: QdrantClient
    _embedder: Embedder
    _db: PersistentDatabase
    _served: bool
    _chunker: Chunker
    _lock: threading.Lock
    _rag_ids: dict[UUID, UUID]

    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        db: PersistentDatabase,
        *,
        served: bool = False,
    ) -> None:
        """Keep the Qdrant `client` shared by every collection, the `embedder` and the `db` holding the links.

        `served` tells whether `client` reaches a Qdrant server, rather than an embedded one.
        """
        self._client = client
        self._embedder = embedder
        self._db = db
        self._served = served
        self._chunker = Chunker(embedder)
        self._lock = threading.Lock()
        self._rag_ids = {}

    async def index(self, rag_id: UUID, source: str, text: str) -> int:
        """Chunk, embed and store `text`, read from `source`, in the RAG `rag_id`, and return the chunk count.

        Re-indexing the same source replaces what was stored for it. The collection is checked for the right
        vector size first, since a refusal must leave what is stored untouched rather than drop it and then fail
        to write.

        Raises:
            RagError: If `text` holds nothing once cleaned.
            EmbeddingError: If the chunks could not be embedded.
            VectorStoreError: If the vectors do not fit the collection.

        """
        chunks = await asyncio.to_thread(self._chunker.to_chunks, text, source)
        if not chunks:
            msg = f"no text could be extracted from {source}"
            raise RagError(msg)
        vectors = await asyncio.to_thread(self._chunker.vectorize_batch, chunks)
        points = [
            chunk_point(source, position, chunk, vector)
            for position, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        return await self._locked(self._replace, self._collection(rag_id), source, points, len(vectors[0]))

    async def research(self, rag_id: UUID, question: str) -> list[dict]:
        """Return the chunks of the RAG `rag_id` closest to `question`, best first, each with its `score`.

        A RAG nothing was indexed in yet gives an empty list.

        Raises:
            EmbeddingError: If the question could not be embedded.
            VectorStoreError: If the question's vector does not fit the collection.

        """
        [vector] = await asyncio.to_thread(self._embedder.embed, [question])
        collection = self._collection(rag_id)
        hits = await self._locked(collection.query, vector, SEARCH_LIMIT, MIN_SCORE)
        return [as_chunk(hit) for hit in hits]

    # Lifecycle, by branch

    async def rag_id_for(self, branch: UUID) -> UUID:
        """Return the RAG id of `branch`, linking it to a new one if it has none yet.

        Raises:
            InvalidInstanceError: If no branch has this uuid.
            DatabaseError: If the new link could not be stored.

        """
        rag_id = await self._linked(branch)
        if rag_id is None:
            rag_id = uuid4()
            await self.link(branch, rag_id)
        return rag_id

    async def link(self, branch: UUID, rag_id: UUID) -> None:
        """Link `branch` to the RAG `rag_id`.

        Raises:
            InvalidInstanceError: If no branch has this uuid.
            DatabaseError: If the link could not be stored.

        """
        await asyncio.to_thread(self._db.set_branch_rag_id, branch, rag_id)
        self._rag_ids[branch] = rag_id

    async def fork(self, parent: UUID, child: UUID) -> UUID:
        """Give `child` a new RAG holding a copy of the points of `parent`'s, and return its id.

        Nothing is embedded again. A parent with no RAG gives the child an empty one.

        Raises:
            InvalidInstanceError: If `parent` or `child` is not a branch.
            DatabaseError: If the link could not be stored.
            VectorStoreError: If the points could not be copied.

        """
        new = uuid4()
        source = await self._linked(parent)
        if source is not None:
            await self._locked(self._collection(source).copy_to, self._collection(new))
        await self.link(child, new)
        return new

    async def drop_branch(self, branch: UUID) -> None:
        """Delete the collection of `branch`, if it has one, and forget the link.

        Call it before the branch is removed from the database, which deletes the row holding the id.

        Raises:
            InvalidInstanceError: If no branch has this uuid.
            VectorStoreError: If Qdrant fails to delete the collection.

        """
        rag_id = await self._linked(branch)
        if rag_id is None:
            return
        await self._locked(self._collection(rag_id).drop)
        self._rag_ids.pop(branch, None)

    async def release(self, rag_id: UUID) -> None:
        """Delete the collection of `rag_id` if no branch is linked to it, and leave it alone otherwise.

        Every link goes through `link`, so a RAG id a session holds and that is linked is always in the cache.

        Raises:
            VectorStoreError: If Qdrant fails to delete the collection.

        """
        if rag_id in self._rag_ids.values():
            return
        await self._locked(self._collection(rag_id).drop)

    @staticmethod
    def sources(chunks: list[dict]) -> list[str]:
        """Name the documents the chunks came from, best match first, each once."""
        seen: list[str] = []
        for chunk in chunks:
            source = str(chunk["source"])
            if source not in seen:
                seen.append(source)
        return seen

    def _collection(self, rag_id: UUID) -> Collection:
        """Reach the collection of `rag_id` through the shared client."""
        return Collection(self._client, rag_id, served=self._served)

    async def _linked(self, branch: UUID) -> UUID | None:
        """Return the RAG id `branch` is linked to, from the cache then the database, or None if it has none.

        Raises:
            InvalidInstanceError: If no branch has this uuid.

        """
        rag_id = self._rag_ids.get(branch)
        if rag_id is None:
            rag_id = await asyncio.to_thread(self._db.get_branch_rag_id, branch)
            if rag_id is not None:
                self._rag_ids[branch] = rag_id
        return rag_id

    async def _locked[T](self, call: Callable[..., T], *args: object) -> T:
        """Run the Qdrant `call` in a thread, holding the lock serialising every Qdrant call."""

        def run() -> T:
            with self._lock:
                return call(*args)

        return await asyncio.to_thread(run)

    @staticmethod
    def _replace(collection: Collection, source: str, points: list[PointStruct], dimension: int) -> int:
        """Replace what `collection` stores for `source` by `points`, and return how many were written."""
        collection.ensure(dimension)
        collection.delete(by_source(source))
        return collection.upsert(points)
