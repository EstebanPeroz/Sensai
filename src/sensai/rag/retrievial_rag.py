"""Answering a question from the chunks the indexation stored."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from sensai.llm.message import Role
from sensai.rag.collection import DEFAULT_LIMIT, as_chunk

if TYPE_CHECKING:
    from sensai.llm.responses import ChatResponse
    from sensai.rag.collection import Collection
    from sensai.rag.indexation_rag import Indexation

DEFAULT_MIN_SCORE = 0.45


_NOTHING_FOUND = "The indexed documents hold nothing close enough to that question."
_GROUNDING = (
    "Answer the question using only the extracts below. If they do not hold the "
    "answer, say so plainly instead of guessing. Name the sources you used."
)


class Answerer(Protocol):
    """The only thing the retrieval needs from a provider: answering one prompt."""

    def current_model(self) -> str:
        """Name of the model that will answer."""
        ...

    def chat(self, payload: dict, *, stream: Literal[False]) -> ChatResponse:
        """Answer the messages of `payload` in one go."""
        ...


class Retrieval:
    """Answer a question from the chunks stored by the indexation."""

    _indexation: Indexation
    _collection: Collection
    _answerer: Answerer

    def __init__(self, indexation: Indexation, collection: Collection, answerer: Answerer) -> None:
        """Keep the pipeline that encodes the question, the collection, and the model that answers."""
        self._indexation = indexation
        self._collection = collection
        self._answerer = answerer

    def answer(
        self,
        question: str,
        limit: int = DEFAULT_LIMIT,
        min_score: float | None = DEFAULT_MIN_SCORE,
    ) -> tuple[str, list[str]]:
        """Answer `question` from the indexed documents, and name the sources used.

        Nothing close enough found means no call to the model at all: there would
        be no grounding for it to answer from.

        Raises:
            EmbeddingError: If the question could not be embedded.
            VectorStoreError: If the question's vector does not fit the collection.
            SensaiError: If the model could not be reached.

        """
        chunks = self.research(question, limit=limit, min_score=min_score)
        if not chunks:
            return _NOTHING_FOUND, []
        return self.generation(self.create_prompt(question, chunks)), self.get_sources(chunks)

    def _encode(self, text: str) -> list[float]:
        """Turn the question into a vector, with the model that indexed the chunks.

        This is the very method the indexation used, so the question and the
        chunks land in the same space: vectors from two models are not comparable.

        Raises:
            EmbeddingError: If the provider could not embed the question.

        """
        return self._indexation.vectorize(text)

    def research(
        self,
        question: str,
        limit: int = DEFAULT_LIMIT,
        min_score: float | None = DEFAULT_MIN_SCORE,
    ) -> list[dict]:
        """Return the stored chunks closest to `question`, best first, each with its `score`.

        Raises:
            EmbeddingError: If the question could not be embedded.
            VectorStoreError: If the question's vector does not fit the collection.

        """
        hits = self._collection.query(self._encode(question), limit=limit, min_score=min_score)
        return [as_chunk(hit) for hit in hits]

    def create_prompt(self, question: str, chunks: list[dict]) -> list[dict]:
        """Build the messages carrying the retrieved chunks and the question.

        The extracts go in a system message, numbered and labelled with their
        source, so the model can ground its answer and cite where it read it.
        """
        extracts = "\n\n".join(
            f"[{number}] {chunk['source']} (passage {chunk['position']})\n{chunk['text']}"
            for number, chunk in enumerate(chunks, start=1)
        )
        return [
            {"role": Role.SYSTEM.value, "content": f"{_GROUNDING}\n\n{extracts}"},
            {"role": Role.USER.value, "content": question},
        ]

    def generation(self, messages: list[dict]) -> str:
        """Ask the model to answer from `messages`, and return what it said.

        Raises:
            SensaiError: If the model could not be reached or answered an error.

        """
        payload = {"model": self._answerer.current_model(), "messages": messages}
        response = self._answerer.chat(payload, stream=False)
        if response.error is not None:
            return f"The model answered an error: {response.error}"
        return response.content or ""

    def get_sources(self, chunks: list[dict]) -> list[str]:
        """Name the documents the retrieved chunks came from, best match first."""
        seen: list[str] = []
        for chunk in chunks:
            source = str(chunk["source"])
            if source not in seen:
                seen.append(source)
        return seen
