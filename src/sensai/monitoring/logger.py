from dataclasses import replace
from typing import TYPE_CHECKING

from sensai.monitoring.interaction import Interaction

if TYPE_CHECKING:
    from sensai.config.settings import LoggingSettings
    from sensai.core.events import EventBus
    from sensai.memory.persistent_db import PersistentDatabase


class InteractionLogger:
    """Observer of the `EventBus` writing every emitted `Interaction` to the persistent database."""

    _db: PersistentDatabase
    _settings: LoggingSettings

    def __init__(self, db: PersistentDatabase, settings: LoggingSettings) -> None:
        """Store `interaction`s in `db`, as configured by `settings`."""
        self._db = db
        self._settings = settings

    def attach(self, bus: EventBus) -> None:
        """Log every interaction emitted on `bus` from now on; does nothing if logging is disabled."""
        if self._settings.enabled:
            bus.subscribe(Interaction, self.log)

    def log(self, interaction: Interaction) -> None:
        """Store `interaction`, stripped of its content unless `log_content` is set.

        Raises:
            DatabaseError: If it could not be stored.

        """
        if not self._settings.log_content:
            interaction = self._without_content(interaction)
        self._db.add_interaction(interaction)

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
