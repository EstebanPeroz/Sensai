"""The Qdrant collection holding the chunks of one RAG id, and the shape of what is stored in it.

The client is shared by every collection: the `Rag` opens it once with `open_client`, and builds a cheap
`Collection` per call. It stays sync, so async callers run it in a thread (`asyncio.to_thread`).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from qdrant_client.http.exceptions import ApiException
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

from sensai_server.rag.error import VectorStoreError

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Record, ScoredPoint

DEFAULT_LIMIT = 5
_SCROLL_PAGE = 256
# Namespace the chunk ids are derived in, so that the same passage of the same
# document always lands on the same point. Any fixed UUID would do; this one
# just has to never change, or every stored chunk becomes unreachable by id.
_CHUNK_NAMESPACE = uuid.UUID("6f0b6e1a-3e4e-4b7a-9c2f-8d1a5b4c7e90")


def chunk_point(source: str, position: int, text: str, vector: list[float]) -> PointStruct:
    """Build the point storing one chunk, under an id derived from its source and position.

    The id is derived rather than random, so re-indexing a document writes over
    its own chunks instead of landing beside them. It does not make the delete
    before a re-index pointless: a document that lost passages would leave the
    extra ones behind, since nothing writes over them.

    The id is only unique inside a collection, which is enough: each RAG id has its own.
    """
    return PointStruct(
        id=str(uuid.uuid5(_CHUNK_NAMESPACE, f"{source}#{position}")),
        vector=vector,
        payload={
            "source": source,
            "position": position,
            "text": text,
        },
    )


def _vector_size(point: PointStruct) -> int:
    """How many dimensions `point` carries.

    A PointStruct accepts far more than a list of floats: named vectors, sparse
    ones, and the Document/Image objects the server embeds itself. None of those
    has a length, so the ones this collection does not store are refused here
    rather than failing on a len() further down.

    Raises:
        VectorStoreError: If the point does not carry a plain list of floats.

    """
    if not isinstance(point.vector, list):
        msg = f"the point {point.id} does not carry a plain list of floats"
        raise VectorStoreError(msg)
    return len(point.vector)


def _as_point(record: Record) -> PointStruct:
    """Turn a record read back with its vector into the point that writes it again.

    Raises:
        VectorStoreError: If the record does not carry a plain list of floats.

    """
    if not isinstance(record.vector, list):
        msg = f"the point {record.id} does not carry a plain list of floats"
        raise VectorStoreError(msg)
    return PointStruct(id=record.id, vector=record.vector, payload=record.payload)


def by_source(source: str) -> Filter:
    """Match the points stored for `source`."""
    return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])


def as_chunk(point: ScoredPoint) -> dict:
    """Turn a hit into its stored payload, with the score added."""
    return {**(point.payload or {}), "score": point.score}


class Collection:
    """The Qdrant collection of one RAG id: the client reaching it, its name, and its vector size.

    Holds nothing about sources or chunks: what a payload carries is decided by
    whoever builds the points, and the helpers above are where that shape lives.
    It never closes the client, which every other collection shares.
    """

    _client: QdrantClient
    _name: str
    _served: bool

    def __init__(self, client: QdrantClient, rag_id: uuid.UUID, *, served: bool = False) -> None:
        """Reach the collection of `rag_id` through `client`, `served` telling whether it is a Qdrant server."""
        self._client = client
        self._name = str(rag_id)
        self._served = served

    @property
    def name(self) -> str:
        """Name of the collection written to and read from: its RAG id."""
        return self._name

    def exists(self) -> bool:
        """Whether the collection has been created yet."""
        try:
            return self._client.collection_exists(self._name)
        except ApiException as error:
            msg = f"could not check whether the collection {self._name!r} exists: {error}"
            raise VectorStoreError(msg) from error

    def dimension(self) -> int | None:
        """Vector size the collection holds, or None while it does not exist yet."""
        if not self.exists():
            return None
        config = self._client.get_collection(self._name).config.params.vectors
        if not isinstance(config, VectorParams):
            msg = f"the collection {self._name!r} holds named vectors, which this one does not use"
            raise VectorStoreError(msg)
        return config.size

    def ensure(self, dimension: int) -> None:
        """Create the collection at `dimension`, or check that it already holds vectors of that size.

        Raises:
            VectorStoreError: If the collection holds vectors of another size.

        """
        stored = self.dimension()
        if stored is None:
            self._client.create_collection(
                self._name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            if self._served:
                self._client.create_payload_index(
                    self._name,
                    field_name="source",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            return
        if stored != dimension:
            msg = (
                f"the collection {self._name!r} holds {stored}-dimension vectors, not {dimension}: "
                f"re-index every source after changing the embedding model"
            )
            raise VectorStoreError(msg)

    def upsert(self, points: list[PointStruct]) -> int:
        """Add or update the points in the collection, and return how many were written.

        Raises:
            VectorStoreError: If the points have different sizes, or a size the collection does not hold.

        """
        if not points:
            return 0
        dimensions = {_vector_size(point) for point in points}
        if len(dimensions) > 1:
            msg = f"inconsistent dimensions in the batch: {sorted(dimensions)}"
            raise VectorStoreError(msg)
        self.ensure(dimensions.pop())
        self._client.upsert(self._name, points=points, wait=True)
        return len(points)

    def query(
        self,
        vector: list[float],
        limit: int = DEFAULT_LIMIT,
        min_score: float | None = None,
        where: Filter | None = None,
    ) -> list[ScoredPoint]:
        """Find the points closest to `vector`, among the ones `where` matches.

        Raises:
            VectorStoreError: If `vector` has a size the collection does not hold, or Qdrant fails.

        """
        stored = self.dimension()
        if stored is None:
            return []
        if stored != len(vector):
            msg = f"searching with a {len(vector)}-dimension vector, but the collection holds {stored}"
            raise VectorStoreError(msg)
        try:
            return self._client.query_points(
                self._name,
                query=vector,
                query_filter=where,
                limit=limit,
                score_threshold=min_score,
                with_payload=True,
            ).points
        except ApiException as error:
            msg = f"could not search the collection {self._name!r}: {error}"
            raise VectorStoreError(msg) from error

    def delete(self, where: Filter) -> None:
        """Delete the points `where` matches."""
        if not self.exists():
            return
        self._client.delete(self._name, points_selector=FilterSelector(filter=where))

    def drop(self) -> None:
        """Delete the whole collection, if it exists.

        Raises:
            VectorStoreError: If Qdrant fails to delete it.

        """
        if not self.exists():
            return
        try:
            self._client.delete_collection(self._name)
        except ApiException as error:
            msg = f"could not delete the collection {self._name!r}: {error}"
            raise VectorStoreError(msg) from error

    def copy_to(self, target: Collection) -> int:
        """Copy every point, vectors included, into `target` under the same ids, and return how many were copied.

        Nothing is embedded again. Nothing to copy creates nothing in `target`.

        Raises:
            VectorStoreError: If `target` holds vectors of another size.

        """
        dimension = self.dimension()
        if dimension is None:
            return 0
        copied = 0
        offset = None
        while True:
            records, offset = self._client.scroll(
                self._name,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if records and not copied:
                target.ensure(dimension)
            copied += target.upsert([_as_point(record) for record in records])
            if offset is None:
                return copied

    def count(self, where: Filter | None = None) -> int:
        """How many points `where` matches, or the whole collection holds."""
        if not self.exists():
            return 0
        return self._client.count(self._name, count_filter=where).count

    def payloads(self, where: Filter | None = None) -> list[dict]:
        """Read the payloads of the points `where` matches, without their vectors."""
        if not self.exists():
            return []
        found: list[dict] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                self._name,
                scroll_filter=where,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            found.extend(point.payload for point in points if point.payload)
            if offset is None:
                return found
