from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from sensai_server.monitoring.interaction import Interaction

if TYPE_CHECKING:
    from sensai_server.config.settings import LoggingSettings
    from sensai_server.events import EventBus
    from sensai_server.monitoring.interaction_repository import InteractionRepository

logger = logging.getLogger(__name__)


class InteractionLogger:
    """Observer of the `EventBus` writing every emitted `Interaction` to an `InteractionRepository`.

    `log` runs on the event loop that every session shares, often in the `finally` of a closing
    stream: it never writes inline. Each write runs in a thread, as a background task, and a failed
    write is only logged on the server. Call `drain` at shutdown so the pending writes are not lost.
    """

    _repository: InteractionRepository
    _settings: LoggingSettings
    _pending: set[asyncio.Task[None]]

    def __init__(self, repository: InteractionRepository, settings: LoggingSettings) -> None:
        """Store `interaction`s in `repository`, as configured by `settings`."""
        self._repository = repository
        self._settings = settings
        self._pending = set()

    def attach(self, bus: EventBus) -> None:
        """Log every interaction emitted on `bus` from now on; does nothing if logging is disabled."""
        if self._settings.enabled:
            bus.subscribe(Interaction, self.log)

    def log(self, interaction: Interaction) -> None:
        """Schedule the storage of `interaction`, stripped of its content unless `log_content` is set.

        Must be called from the running event loop. Returns at once: the write happens in a thread.
        """
        if not self._settings.log_content:
            interaction = self._without_content(interaction)
        task = asyncio.get_running_loop().create_task(asyncio.to_thread(self._repository.add, interaction))
        self._pending.add(task)
        task.add_done_callback(self._on_written)

    async def drain(self) -> None:
        """Wait until every scheduled write, including those scheduled meanwhile, is done."""
        while self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)

    def _on_written(self, task: asyncio.Task[None]) -> None:
        """Release `task` and log its failure, if any: a failed write never reaches the chat."""
        self._pending.discard(task)
        if task.cancelled():
            logger.warning("An interaction write was cancelled")
        elif (error := task.exception()) is not None:
            logger.warning("Failed to store an interaction: %s", error, exc_info=error)

    @staticmethod
    def _without_content(interaction: Interaction) -> Interaction:
        """Keep the metrics and the names of the called tools, drop what the user and the model wrote."""
        return replace(
            interaction,
            request=None,
            response=None,
            thinking=None,
            tool_calls=[{"name": (call.get("function") or {}).get("name")} for call in interaction.tool_calls],
        )
