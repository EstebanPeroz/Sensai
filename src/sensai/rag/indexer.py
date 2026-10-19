"""Wire the document pipeline to the vector store."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sensai.rag.error import RagError
from sensai.rag.indexation_rag import ChunkStrategy, Indexation

if TYPE_CHECKING:
    from sensai.rag.store import VectorStore


class Indexer:
    """Run a document through the pipeline and store the result."""

    def __init__(self, indexation: Indexation, store: VectorStore) -> None:
        """Keep the pipeline and the store this indexer writes to."""
        self._indexation = indexation
        self._store = store

    def index(self, source: str, *, strategy: ChunkStrategy = ChunkStrategy.AUTO) -> int:
        """Collect, clean, chunk, embed and store `source`, and return the chunk count.

        Re-indexing the same source replaces what was stored for it.

        Raises:
            RagError: If no text could be extracted from `source`.

        """
        chunks = self._indexation.to_chunks(source, strategy=strategy)
        if not chunks:
            msg = f"no text could be extracted from {source}"
            raise RagError(msg)
        vectors = self._indexation.vectorize_batch(chunks)
        return self._store.add_chunks(source, chunks, vectors, self._indexation.model)
