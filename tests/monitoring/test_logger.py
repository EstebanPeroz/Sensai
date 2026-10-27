from unittest.mock import MagicMock

from sensai.config.settings import LoggingSettings
from sensai.core.events import EventBus
from sensai.monitoring.interaction import Interaction, InteractionStatus
from sensai.monitoring.interaction_repository import InteractionRepository
from sensai.monitoring.logger import InteractionLogger

_INTERACTION = Interaction(
    model="m",
    status=InteractionStatus.OK,
    latency_ms=42.0,
    request=[{"role": "user", "content": "my password is hunter2"}],
    response="noted",
    thinking="the user shared a secret",
    tool_calls=[{"function": {"name": "web_search", "arguments": {"query": "hunter2"}}}],
)


def attached(settings: LoggingSettings) -> tuple[MagicMock, EventBus]:
    repository, bus = MagicMock(spec=InteractionRepository), EventBus()
    InteractionLogger(repository, settings).attach(bus)
    return repository, bus


class TestInteractionLogger:
    def test_stores_interactions_emitted_on_the_bus(self) -> None:
        repository, bus = attached(LoggingSettings())

        bus.emit(_INTERACTION)
        bus.emit("not an interaction")

        repository.add.assert_called_once_with(_INTERACTION)

    def test_does_nothing_when_disabled(self) -> None:
        repository, bus = attached(LoggingSettings(enabled=False))

        bus.emit(_INTERACTION)

        repository.add.assert_not_called()

    def test_strips_content_but_keeps_metrics_and_tool_names(self) -> None:
        repository, bus = attached(LoggingSettings(log_content=False))

        bus.emit(_INTERACTION)

        stored = repository.add.call_args.args[0]
        assert (stored.request, stored.response, stored.thinking) == (None, None, None)
        assert stored.tool_calls == [{"name": "web_search"}]
        assert (stored.id, stored.latency_ms) == (_INTERACTION.id, _INTERACTION.latency_ms)
