"""Wire the document pipeline to the Qdrant collection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sensai.rag.collection import by_source, chunk_point
from sensai.rag.error import RagError
from sensai.rag.indexation_rag import ChunkStrategy

if TYPE_CHECKING:
    from sensai.rag.collection import Collection
    from sensai.rag.indexation_rag import Indexation


class Indexer:
    """Run a document through the pipeline and store the result."""

    _indexation: Indexation
    _collection: Collection

    def __init__(self, indexation: Indexation, collection: Collection) -> None:
        """Keep the pipeline and the collection this indexer writes to."""
        self._indexation = indexation
        self._collection = collection

    def index(self, source: str, *, strategy: ChunkStrategy = ChunkStrategy.AUTO) -> int:
        """Collect, clean, chunk, embed and store `source`, and return the chunk count.

        Re-indexing the same source replaces what was stored for it: the chunks
        are dropped before the new ones are written, so a document that lost a
        passage does not keep it. The collection is checked for the right vector
        size first, since a refusal must leave what is stored untouched rather
        than drop it and then fail to write.

        Raises:
            RagError: If no text could be extracted from `source`.
            EmbeddingError: If the chunks could not be embedded.
            VectorStoreError: If the vectors do not fit the collection.

        """
        chunks = self._indexation.to_chunks(source, strategy=strategy)
        if not chunks:
            msg = f"no text could be extracted from {source}"
            raise RagError(msg)
        vectors = self._indexation.vectorize_batch(chunks)
        points = []
        for position, (text, vector) in enumerate(zip(chunks, vectors, strict=True)):
            points.append(chunk_point(source, position, text, vector, self._indexation.model))
        self._collection.ensure(len(vectors[0]))
        self._collection.delete(by_source(source))
        return self._collection.upsert(points)
