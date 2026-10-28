"""The Qdrant collection holding the RAG chunks, and the shape of what is stored in it."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
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

from sensai.rag.error import VectorStoreError

if TYPE_CHECKING:
    from qdrant_client.models import ScoredPoint

DEFAULT_LOCATION = "http://localhost:6333"
DEFAULT_COLLECTION = "documentation"
DEFAULT_LIMIT = 5
_SCROLL_PAGE = 256
# Namespace the chunk ids are derived in, so that the same passage of the same
# document always lands on the same point. Any fixed UUID would do; this one
# just has to never change, or every stored chunk becomes unreachable by id.
_CHUNK_NAMESPACE = uuid.UUID("6f0b6e1a-3e4e-4b7a-9c2f-8d1a5b4c7e90")


def open_client(location: str = DEFAULT_LOCATION) -> QdrantClient:
    """Open the client `location` names: a server URL, ":memory:", or a directory.

    A bare directory has to go through `path=`: passed positionally, the client
    would read it as a URL and fail on the first request instead of here.
    """
    if location == ":memory:" or location.startswith(("http://", "https://")):
        return QdrantClient(location)
    return QdrantClient(path=location)


def chunk_point(source: str, position: int, text: str, vector: list[float]) -> PointStruct:
    """Build the point storing one chunk, under an id derived from its source and position.

    The id is derived rather than random, so re-indexing a document writes over
    its own chunks instead of landing beside them. It does not make the delete
    before a re-index pointless: a document that lost passages would leave the
    extra ones behind, since nothing writes over them.
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


def by_source(source: str) -> Filter:
    """Match the points stored for `source`."""
    return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])


def as_chunk(point: ScoredPoint) -> dict:
    """Turn a hit into its stored payload, with the score added."""
    return {**(point.payload or {}), "score": point.score}


class Collection:
    """One Qdrant collection: the client reaching it, its name, and its vector size.

    Holds nothing about sources or chunks: what a payload carries is decided by
    whoever builds the points, and the helpers above are where that shape lives.
    """

    _client: QdrantClient
    _name: str
    _served: bool

    @classmethod
    def open(cls, location: str = DEFAULT_LOCATION, name: str = DEFAULT_COLLECTION) -> Collection:
        """Open the collection `name` on the Qdrant `location` names."""
        served = location.startswith(("http://", "https://"))
        return cls(open_client(location), name, served=served)

    def __init__(self, client: QdrantClient, name: str = DEFAULT_COLLECTION, *, served: bool = False) -> None:
        """Reach the collection `name` through `client`, `served` telling whether it is a Qdrant server."""
        self._client = client
        self._name = name
        self._served = served

    @property
    def name(self) -> str:
        """Name of the collection written to and read from."""
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
        """Check if the dimension used by the next vectors."""
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
                f"the col lection {self._name!r} holds {stored}-dimension vectors, not {dimension}: "
                f"re-index every source after changing the embedding model"
            )
            raise VectorStoreError(msg)

    def upsert(self, points: list[PointStruct]) -> int:
        """Add or update the points in the collection.

        There are two errors possible,
        txo different dimenseion in the collection or the one checked by ensure.
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
    ) -> list[ScoredPoint]:
        """Research the points wo are usefull to the vector given."""
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
                limit=limit,
                score_threshold=min_score,
                with_payload=True,
            ).points
        except ApiException as error:
            msg = f"could not search the collection {self._name!r}: {error}"
            raise VectorStoreError(msg) from error

    def delete(self, where: Filter) -> None:
        """Delete a point."""
        if not self.exists():
            return
        self._client.delete(self._name, points_selector=FilterSelector(filter=where))

    def count(self, where: Filter | None = None) -> int:
        """Only used in the test, to check the nb of point."""
        if not self.exists():
            return 0
        return self._client.count(self._name, count_filter=where).count

    def payloads(self, where: Filter | None = None) -> list[dict]:
        """Research about the paylod especially and not the vectors."""
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

    def close(self) -> None:
        """Release the client, which an embedded database keeps locked for one process."""
        self._client.close()
