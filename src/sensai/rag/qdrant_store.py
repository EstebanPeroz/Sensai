from __future__ import annotations

import uuid
from datetime import UTC, datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from sensai.rag.error import VectorStoreError

DEFAULT_LOCATION = "rag_qdrant"
DEFAULT_COLLECTION = "documentation"
DEFAULT_LIMIT = 5
_SCROLL_PAGE = 256


class QdrantStore:
    """Store the embedded chunks in a Qdrant collection and find them again."""

    _client: QdrantClient
    _collection: str
    _served: bool

    def __init__(self, location: str = DEFAULT_LOCATION, collection: str = DEFAULT_COLLECTION) -> None:
        """Open `location`: a server URL, ":memory:", or a directory holding an embedded database.

        The collection is not created here: its vector size is only known once
        the first embeddings arrive, and creating it from a guessed size is what
        makes a change of embedding model fail in confusing ways.
        """
        self._collection = collection
        self._served = location.startswith(("http://", "https://"))
        self._client = QdrantClient(location) if self._served or location == ":memory:" else QdrantClient(path=location)

    def add_chunks(
        self,
        source: str,
        chunks: list[str],
        vectors: list[list[float]],
        model: str,
        *,
        replace: bool = True,
    ) -> int:
        """Store the chunks of `source` with their vectors, and return how many were written.

        If `replace`, the chunks already stored for that source are deleted
        first: re-indexing a document updates it instead of duplicating it.
        Otherwise the chunks are appended, their positions carrying on after
        the highest one already stored for that source.
        An empty `chunks` list writes nothing and deletes nothing.

        Raises:
            VectorStoreError: If `chunks` and `vectors` have different lengths,
                if the vectors do not all share one dimension, or if that
                dimension is not the one the collection already holds.

        """
        if len(chunks) != len(vectors):
            msg = f"{len(chunks)} chunks for {len(vectors)} vectors"
            raise VectorStoreError(msg)
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) > 1:
            msg = f"inconsistent dimensions in the batch: {sorted(dimensions)}"
            raise VectorStoreError(msg)
        if not chunks:
            return 0

        self._ensure_collection(dimensions.pop())
        if replace:
            self._client.delete(self._collection, points_selector=FilterSelector(filter=self._by_source(source)))
        start = self._next_position(source)
        stamped = datetime.now(tz=UTC).isoformat()
        self._client.upsert(
            self._collection,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "source": source,
                        "position": start + offset,
                        "text": text,
                        "model": model,
                        "created_at": stamped,
                    },
                )
                for offset, (text, vector) in enumerate(zip(chunks, vectors, strict=True))
            ],
        )
        return len(chunks)

    def search(self, vector: list[float], limit: int = DEFAULT_LIMIT, min_score: float | None = None) -> list[dict]:
        """Return the stored chunks closest to `vector`, best first, each with its `score`.

        `min_score` drops the chunks that are not close enough. Without it, a
        question about nothing in the collection still comes back with `limit`
        chunks, and whatever reads them cannot tell that none of them is
        relevant.

        Raises:
            VectorStoreError: If the vector is not the size the collection holds.

        """
        stored = self.dimension()
        if stored is None:
            return []
        if stored != len(vector):
            msg = f"searching with a {len(vector)}-dimension vector, but the collection holds {stored}"
            raise VectorStoreError(msg)
        answer = self._client.query_points(
            self._collection,
            query=vector,
            limit=limit,
            score_threshold=min_score,
            with_payload=True,
        )
        return [{**point.payload, "score": point.score} for point in answer.points if point.payload]

    def remove_source(self, source: str) -> int:
        """Drop every chunk stored for `source`, and return how many were dropped."""
        removed = self.count(source)
        if removed:
            self._client.delete(self._collection, points_selector=FilterSelector(filter=self._by_source(source)))
        return removed

    def count(self, source: str | None = None) -> int:
        """How many chunks are stored, in total or for one source."""
        if not self._client.collection_exists(self._collection):
            return 0
        condition = None if source is None else self._by_source(source)
        return self._client.count(self._collection, count_filter=condition).count

    def sources(self) -> set[str]:
        """Every source that has at least one chunk stored."""
        return {str(payload["source"]) for payload in self._payloads(None)}

    def dimension(self) -> int | None:
        """Vector size the collection holds, or None while it does not exist yet."""
        if not self._client.collection_exists(self._collection):
            return None
        return int(self._client.get_collection(self._collection).config.params.vectors.size)

    def close(self) -> None:
        """Release the database, which an embedded one keeps locked for one process."""
        self._client.close()

    def _ensure_collection(self, dimension: int) -> None:
        """Create the collection on first use, and refuse vectors of another size.

        Raises:
            VectorStoreError: If the collection already holds vectors of a
                different size, which changing the embedding model is enough to
                cause: the stored vectors would not be comparable to the new ones.

        """
        stored = self.dimension()
        if stored is None:
            self._client.create_collection(
                self._collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            if self._served:
                self._client.create_payload_index(
                    self._collection,
                    field_name="source",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            return
        if stored != dimension:
            msg = (
                f"the collection {self._collection!r} holds {stored}-dimension vectors, not {dimension}: "
                f"re-index every source after changing the embedding model"
            )
            raise VectorStoreError(msg)

    def _next_position(self, source: str) -> int:
        """First free position for `source`: 0 if it has no stored chunk."""
        positions = [int(payload["position"]) for payload in self._payloads(source)]
        return max(positions) + 1 if positions else 0

    def _payloads(self, source: str | None) -> list[dict]:
        """Every stored payload, for one source or for all of them."""
        if not self._client.collection_exists(self._collection):
            return []
        condition = None if source is None else self._by_source(source)
        found: list[dict] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                self._collection,
                scroll_filter=condition,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            found.extend(point.payload for point in points if point.payload)
            if offset is None:
                return found

    @staticmethod
    def _by_source(source: str) -> Filter:
        """Match the points stored for `source`."""
        return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
