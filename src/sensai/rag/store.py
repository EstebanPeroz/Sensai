"""Storage of the chunks and their embeddings for the RAG."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Engine, create_engine, delete, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from sensai.rag.error import VectorStoreError


def dump_vector(vector: list[float]) -> str:
    """Serialise an embedding for storage."""
    return json.dumps(vector)


def load_vector(raw: str) -> list[float]:
    """Deserialise an embedding read from the database."""
    return json.loads(raw)


class Base(DeclarativeBase):
    """Base of the RAG tables: every row has an id and a creation time."""

    __abstract__ = True

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
    )


class ChunkTable(Base):
    """A piece of a document, with the embedding that matches it.

    `vector` is the embedding serialised to JSON and `model` the model that
    produced it: vectors from different models are not comparable.
    """

    __tablename__ = "chunk"

    source: Mapped[str] = mapped_column(index=True)
    position: Mapped[int] = mapped_column()
    text: Mapped[str] = mapped_column()
    vector: Mapped[str] = mapped_column()
    model: Mapped[str] = mapped_column(index=True)
    dim: Mapped[int] = mapped_column()


class VectorStore:
    """Store the embedded chunks in a SQLite database and retrieve them."""

    _engine: Engine

    def __init__(self, url: str = "sqlite:///rag.db") -> None:
        """Connect to the database at `url` and create the table if it is missing."""
        self._engine = create_engine(url)
        Base.metadata.create_all(self._engine)

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
                if the vectors do not all share one dimension, or if the write fails.

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

        with Session(self._engine) as session:
            if replace:
                session.execute(delete(ChunkTable).where(ChunkTable.source == source))
            for position, (text, vector) in enumerate(
                zip(chunks, vectors, strict=True), start=self._next_position(session, source)
            ):
                session.add(
                    ChunkTable(
                        source=source,
                        position=position,
                        text=text,
                        vector=dump_vector(vector),
                        model=model,
                        dim=len(vector),
                    )
                )
            try:
                session.commit()
            except SQLAlchemyError:
                msg = "the chunks could not be written"
                raise VectorStoreError(msg) from None
        return len(chunks)

    @staticmethod
    def _next_position(session: Session, source: str) -> int:
        """First free position for `source`: 0 if it has no stored chunk."""
        highest = session.scalar(select(func.max(ChunkTable.position)).where(ChunkTable.source == source))
        return 0 if highest is None else highest + 1
