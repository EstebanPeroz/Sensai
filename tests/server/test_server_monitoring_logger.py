import asyncio
import logging
import threading
from unittest.mock import MagicMock

import pytest
from sensai_server.config.settings import LoggingSettings
from sensai_server.events import EventBus
from sensai_server.memory.persistent_db.error import DatabaseError
from sensai_server.monitoring.interaction import Interaction, InteractionStatus
from sensai_server.monitoring.interaction_repository import InteractionRepository
from sensai_server.monitoring.logger import InteractionLogger

_INTERACTION = Interaction(
    model="m",
    status=InteractionStatus.OK,
    latency_ms=42.0,
    request=[{"role": "user", "content": "my password is hunter2"}],
    response="noted",
    thinking="the user shared a secret",
    tool_calls=[{"function": {"name": "web_search", "arguments": {"query": "hunter2"}}}],
)


def attached(settings: LoggingSettings, repository: InteractionRepository) -> tuple[EventBus, InteractionLogger]:
    bus, interaction_logger = EventBus(), InteractionLogger(repository, settings)
    interaction_logger.attach(bus)
    return bus, interaction_logger


class BlockingRepository:
    """A repository whose `add` waits until the test releases it."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.added: list[Interaction] = []

    def add(self, interaction: Interaction) -> None:
        self.release.wait(timeout=5)
        self.added.append(interaction)

    def get_all(self, **_: object) -> list[Interaction]:
        return []


class TestInteractionLogger:
    async def test_stores_interactions_emitted_on_the_bus(self) -> None:
        repository = MagicMock(spec=InteractionRepository)
        bus, interaction_logger = attached(LoggingSettings(), repository)

        bus.emit(_INTERACTION)
        bus.emit("not an interaction")
        await interaction_logger.drain()

        repository.add.assert_called_once_with(_INTERACTION)

    async def test_does_nothing_when_disabled(self) -> None:
        repository = MagicMock(spec=InteractionRepository)
        bus, interaction_logger = attached(LoggingSettings(enabled=False), repository)

        bus.emit(_INTERACTION)
        await interaction_logger.drain()

        repository.add.assert_not_called()

    async def test_strips_content_but_keeps_metrics_and_tool_names(self) -> None:
        repository = MagicMock(spec=InteractionRepository)
        bus, interaction_logger = attached(LoggingSettings(log_content=False), repository)

        bus.emit(_INTERACTION)
        await interaction_logger.drain()

        stored = repository.add.call_args.args[0]
        assert (stored.request, stored.response, stored.thinking) == (None, None, None)
        assert stored.tool_calls == [{"name": "web_search"}]
        assert (stored.id, stored.latency_ms) == (_INTERACTION.id, _INTERACTION.latency_ms)

    async def test_a_slow_repository_does_not_delay_emit(self) -> None:
        repository = BlockingRepository()
        bus, interaction_logger = attached(LoggingSettings(), repository)

        bus.emit(_INTERACTION)  # returns although the write is blocked
        await asyncio.sleep(0)

        assert repository.added == []
        repository.release.set()
        await interaction_logger.drain()
        assert repository.added == [_INTERACTION]

    async def test_drain_waits_for_the_pending_writes(self) -> None:
        repository = BlockingRepository()
        bus, interaction_logger = attached(LoggingSettings(), repository)
        bus.emit(_INTERACTION)
        bus.emit(_INTERACTION)

        drained = asyncio.create_task(interaction_logger.drain())
        await asyncio.sleep(0.05)
        assert not drained.done()

        repository.release.set()
        await asyncio.wait_for(drained, timeout=5)
        assert repository.added == [_INTERACTION, _INTERACTION]

    async def test_drain_returns_at_once_without_pending_writes(self) -> None:
        _, interaction_logger = attached(LoggingSettings(), MagicMock(spec=InteractionRepository))

        await asyncio.wait_for(interaction_logger.drain(), timeout=1)

    async def test_a_failing_repository_is_only_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        repository = MagicMock(spec=InteractionRepository)
        repository.add.side_effect = DatabaseError("disk full")
        bus, interaction_logger = attached(LoggingSettings(), repository)

        with caplog.at_level(logging.WARNING, logger="sensai_server.monitoring.logger"):
            bus.emit(_INTERACTION)  # does not raise
            await interaction_logger.drain()  # nor does draining

        assert "disk full" in caplog.text
