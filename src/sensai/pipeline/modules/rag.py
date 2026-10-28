from __future__ import annotations

from typing import TYPE_CHECKING

from sensai.error import SensaiError
from sensai.llm.message import Role
from sensai.pipeline.chain import answer_chunk
from sensai.rag.collection import DEFAULT_LIMIT
from sensai.rag.retrievial_rag import DEFAULT_MIN_SCORE

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sensai.llm.responses import ChatResponse
    from sensai.pipeline.chain import ChatContext, Next
    from sensai.rag.retrievial_rag import Retrieval

EXTRACTS_INSTRUCTION = (
    "Extracts from the indexed documents are below. Answer from them when they hold "
    "the answer, and name the sources you used. Ignore them when they do not."
)


class RagStage:
    """Put the chunks closest to the question in the payload, then let the model answer from them."""

    def __init__(
        self,
        retrieval: Retrieval,
        limit: int = DEFAULT_LIMIT,
        min_score: float | None = DEFAULT_MIN_SCORE,
    ) -> None:
        """Keep the retrieval to search, and how many chunks to accept and how close they must be."""
        self._retrieval = retrieval
        self._limit = limit
        self._min_score = min_score

    def __call__(self, context: ChatContext, next_stage: Next) -> Iterator[ChatResponse]:
        """Add the extracts the documents hold on the question to the payload, then name their sources.

        The sources are named once the model is done, which only works because the
        stage stays on the line while the answer streams through it.
        """
        chunks = self._research(context)
        if not chunks:
            yield from next_stage(context)
            return
        context.messages.append(self._extracts(chunks))
        yield from next_stage(context)
        yield answer_chunk(f"\n\nSources: {', '.join(self._retrieval.get_sources(chunks))}")

    def _research(self, context: ChatContext) -> list[dict]:
        """Search the documents on the message, reporting a failed search to the UI rather than raising."""
        try:
            return self._retrieval.research(context.message, limit=self._limit, min_score=self._min_score)
        except SensaiError as error:
            context.ui.send_system_message(f"The indexed documents could not be searched: {error.message}")
            return []

    @staticmethod
    def _extracts(chunks: list[dict]) -> dict:
        """Build the message carrying the extracts, numbered and labelled with their source."""
        extracts = "\n\n".join(
            f"[{number}] {chunk['source']} (passage {chunk['position']})\n{chunk['text']}"
            for number, chunk in enumerate(chunks, start=1)
        )
        return {"role": Role.SYSTEM.value, "content": f"{EXTRACTS_INSTRUCTION}\n\n{extracts}"}
