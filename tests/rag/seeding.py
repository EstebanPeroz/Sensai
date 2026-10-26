"""Putting chunks in a collection without going through a document, for the tests that need some there."""

from sensai.rag.collection import Collection, chunk_point
from sensai.rag.indexation_rag import Indexation


def seed(collection: Collection, indexation: Indexation, source: str, *texts: str) -> None:
    """Store `texts` as the chunks of `source`, as the Indexer would, embedding them with `indexation`."""
    vectors = indexation.vectorize_batch(list(texts))
    collection.ensure(len(vectors[0]))
    collection.upsert(
        [
            chunk_point(source, position, text, vector)
            for position, (text, vector) in enumerate(zip(texts, vectors, strict=True))
        ]
    )


def stored(collection: Collection) -> list[dict]:
    """Every stored payload, ordered by source then position."""
    return sorted(collection.payloads(), key=lambda payload: (payload["source"], payload["position"]))
