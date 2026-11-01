from __future__ import annotations

from typing import TYPE_CHECKING, override

from sensai_protocol.error import SensaiError

from sensai_server.commands._base import Command
from sensai_server.rag.sources import collect

if TYPE_CHECKING:
    from uuid import UUID

    from sensai_server.rag import Rag
    from sensai_server.session import Session


class Index(Command):
    """Index command used to add documents to the RAG of the session's conversation.

    It runs as the session's task, like a chat: the session is busy meanwhile, and `/cancel` stops it between two
    sources. A source whose chunks were already being stored still finishes, which is safe: re-indexing it replaces
    it. Nothing is reported after a cancel.
    """

    name = "index"
    help = "Index documents into the RAG, from paths or URLs"
    usage = "/index <path|url> [path|url ...]"

    _rag: Rag

    def __init__(self, rag: Rag) -> None:
        """Init the index command on the `rag` shared by every session."""
        super().__init__()
        self._rag = rag

    @override
    async def execute(self, session: Session, *args: str) -> None:
        """Index every given source, reporting each one then a summary.

        A source that cannot be indexed is reported and skipped: the others are still indexed, so one bad path
        does not waste the whole batch.
        """
        if not args:
            await session.out.send_system_message(f"Usage: {self.usage}")
            return
        try:
            rag_id = await self._rag_id(session)
        except SensaiError as error:
            await session.out.send_system_message(f"The RAG of this conversation is unavailable: {error.message}")
            return
        chunks, failures = 0, 0
        for source in args:
            try:
                text = await collect(source, session.out)
                indexed = await self._rag.index(rag_id, source, text)
            except SensaiError as error:
                failures += 1
                await session.out.send_system_message(f"{source}: {error.message}")
                continue
            chunks += indexed
            await session.out.send_system_message(f"{source}: {indexed} chunks")
        if len(args) > 1 or failures:
            await session.out.send_system_message(self._summary(len(args), chunks, failures))

    async def _rag_id(self, session: Session) -> UUID:
        """Return the RAG id of the session's conversation, linking its branch to a new one if it has none yet.

        Raises:
            SensaiError: If the branch could not be linked.

        """
        conv = session.conv
        if conv.rag_id is None:
            if conv.uuid is None:
                msg = "the conversation has no RAG id"
                raise SensaiError(msg)
            conv.rag_id = await self._rag.rag_id_for(conv.uuid)
        return conv.rag_id

    @staticmethod
    def _summary(sources: int, chunks: int, failures: int) -> str:
        """One line telling how many sources were indexed, and how many were skipped."""
        indexed = sources - failures
        line = f"Indexed {indexed}/{sources} sources into {chunks} chunks"
        return line if not failures else f"{line} ({failures} skipped)"
