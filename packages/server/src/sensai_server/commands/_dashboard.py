import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, override

from sensai_protocol import Period

from sensai_server.commands._base import Command
from sensai_server.memory.persistent_db.error import DatabaseError
from sensai_server.monitoring.metrics import build_report, history_start, to_protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from sensai_server.monitoring.interaction_repository import InteractionRepository
    from sensai_server.session import Session


class Dashboard(Command):
    """Dashboard command class to show the monitoring metrics of the logged model calls.

    The report covers every logged call, across all sessions and branches: a login has no stable identity to filter by.
    """

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
    async def execute(self, session: Session, *args: str) -> None:
        """Build the report of the requested period from the logged interactions and show it."""
        if len(args) > 1:
            await session.out.send_system_message(f"Invalid number of arguments. Usage: {self.usage}")
            return
        value = args[0] if args else Period.DAY.value
        try:
            period = Period(value)
        except ValueError:
            await session.out.send_system_message(f"Unknown period '{value}'. Usage: {self.usage}")
            return
        now = self._clock()
        try:
            interactions = await asyncio.to_thread(
                self._interactions.get_all, since=history_start(period, now), with_content=False
            )
        except DatabaseError as err:
            await session.out.send_system_message(err.message)
            return
        await session.out.show_dashboard(to_protocol(build_report(interactions, period, now)))

    @override
    def completions(self, session: Session, args: list[str]) -> list[str]:
        """Return the periods the dashboard can cover."""
        del session, args
        return [period.value for period in Period]
