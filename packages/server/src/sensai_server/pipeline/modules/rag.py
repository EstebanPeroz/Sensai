from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING

from sensai_protocol.error import SensaiError
from sensai_protocol.server_messages import Role

from sensai_server.llm.responses import ChatResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sensai_server.pipeline.chain import ChatContext, Next
    from sensai_server.rag import Rag

EXTRACTS_INSTRUCTION = (
    "Extracts from the indexed documents are below. Answer from them when they hold "
    "the answer, and name the sources you used. Ignore them when they do not."
)


class RagStage:
    """Put the chunks of the conversation's RAG closest to the question in the payload, then name their sources."""

    _rag: Rag

    def __init__(self, rag: Rag) -> None:
        """Keep the RAG, shared by every session, this stage searches with each exchange's RAG id."""
        self._rag = rag

    async def __call__(self, context: ChatContext, next_stage: Next) -> AsyncGenerator[ChatResponse]:
        """Add the extracts the documents hold on the question to the payload, and stream the answer.

        The sources are sent as a content chunk right before the model's `done` chunk, so the client still gets a
        single `done`, and they are part of the saved answer. They are left out when the answer failed.
        """
        chunks = await self._research(context)
        if chunks:
            context.messages.append(self._extracts(chunks))
        async with aclosing(next_stage(context)) as answer:
            async for chunk in answer:
                if chunk.done and chunks and chunk.error is None:
                    yield _content_chunk(f"\n\nSources: {', '.join(self._rag.sources(chunks))}")
                yield chunk

    async def _research(self, context: ChatContext) -> list[dict]:
        """Search the conversation's documents on the message, reporting a failed search rather than raising.

        A context with no RAG id gives no chunk.
        """
        if context.rag_id is None:
            return []
        try:
            return await self._rag.research(context.rag_id, context.message)
        except SensaiError as error:
            await context.out.send_system_message(f"The indexed documents could not be searched: {error.message}")
            return []

    @staticmethod
    def _extracts(chunks: list[dict]) -> dict:
        """Build the message carrying the extracts, numbered and labelled with their source."""
        extracts = "\n\n".join(
            f"[{number}] {chunk['source']} (passage {chunk['position']})\n{chunk['text']}"
            for number, chunk in enumerate(chunks, start=1)
        )
        return {"role": Role.SYSTEM.value, "content": f"{EXTRACTS_INSTRUCTION}\n\n{extracts}"}


def _content_chunk(content: str) -> ChatResponse:
    """Build a non-final answer chunk carrying `content`."""
    return ChatResponse({"done": False, "message": {"role": "assistant", "content": content}})
