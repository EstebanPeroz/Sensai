from __future__ import annotations

from typing import TYPE_CHECKING, override

from sensai_protocol.error import SensaiError

from sensai_server.chat.usage import send_context_usage
from sensai_server.commands._base import Command

if TYPE_CHECKING:
    from sensai_server.monitoring.budget import ContextBudget
    from sensai_server.pipeline.modules.compression import Compressor
    from sensai_server.session import Session


class Compress(Command):
    """Compress command, summarising the oldest messages of the conversation without waiting for the budget.

    It runs as the session's task, like a chat: `/cancel` stops the summary, which is then recorded as interrupted
    and not stored.
    """

    name = "compress"
    help = "Summarise the oldest messages of the conversation now, to free some of its context."
    usage = "/compress"

    _compressor: Compressor
    _budget: ContextBudget | None

    def __init__(self, compressor: Compressor, budget: ContextBudget | None = None) -> None:
        """Init the command on the `compressor` shared by every session, sending the usage from `budget`, if any."""
        super().__init__()
        self._compressor = compressor
        self._budget = budget

    @override
    async def execute(self, session: Session, *args: str) -> None:
        """Summarise the oldest messages of the session's branch, tell how much it saved, then send the usage.

        The usage sent is an estimate of the shorter context, until the next chat call measures it exactly.
        """
        if args:
            await session.out.send_system_message(f"This command takes no argument. Usage: {self.usage}")
            return
        conv = session.conv
        if conv.uuid is None or conv.model is None:
            await session.out.send_system_message("Nothing to compress: the conversation has not started.")
            return
        persona = conv.persona.name if conv.persona is not None else None
        try:
            compression = await self._compressor.compress(conv.uuid, conv.history.to_json(), conv.model, persona)
        except SensaiError as err:
            await session.out.send_system_message(f"The conversation could not be compressed: {err.message}")
        else:
            if compression is None:
                await session.out.send_system_message(
                    "Nothing to compress: only the last messages are left to summarise."
                )
            else:
                if self._budget is not None:
                    self._budget.compressed(conv.uuid, compression)
                await session.out.send_system_message(compression.describe())
        await send_context_usage(session, self._budget)
