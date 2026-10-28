from datetime import UTC, datetime
from typing import TYPE_CHECKING, override

from sensai.commands.base import Command
from sensai.memory.persistent_db.error import DatabaseError
from sensai.monitoring.metrics import Period, build_report, history_start

if TYPE_CHECKING:
    from collections.abc import Callable

    from sensai.monitoring.interaction_repository import InteractionRepository


class Dashboard(Command):
    """Dashboard command class to show the monitoring metrics of the logged model calls."""

    name = "dashboard"
    help = "Show the monitoring dashboard of the model calls over the last 24h (default), 7d or 30d."
    usage = "/dashboard [24h|7d|30d]"
    _interactions: InteractionRepository
    _clock: Callable[[], datetime]

    def __init__(
        self, interactions: InteractionRepository, clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC)
    ) -> None:
        """Initialize the Dashboard command with the repository of the logged interactions and the current time."""
        super().__init__()
        self._interactions = interactions
        self._clock = clock

    @override
    def execute(self, *args: str) -> None:
        """Build the report of the requested period from the logged interactions and show it."""
        if len(args) > 1:
            self._ui.send_system_message(f"Invalid number of arguments. Usage: {self.usage}")
            return
        try:
            period = Period(args[0]) if args else Period.DAY
        except ValueError:
            self._ui.send_system_message(f"Unknown period '{args[0]}'. Usage: {self.usage}")
            return
        now = self._clock()
        try:
            interactions = self._interactions.get_all(since=history_start(period, now), with_content=False)
        except DatabaseError as err:
            self._ui.send_system_message(err.message)
            return
        self._ui.show_dashboard(build_report(interactions, period, now))

    @override
    def completions(self, args: list[str]) -> list[str]:
        """Return the periods the dashboard can cover."""
        del args
        return [period.value for period in Period]
