from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
import redis

from sensai.config.settings import ContextSettings, FeatureSettings, LLMSettings, OllamaSettings, Settings
from sensai.core.core import Core
from sensai.llm.error import ModelError
from sensai.llm.responses import ChatMetrics
from sensai.mcp.config import MCPServerConfig
from sensai.mcp.discovery import MCPToolInfo
from sensai.mcp.errors import MCPStartupError
from sensai.mcp.tool import MCPTool
from sensai.memory.persistent_db.error import DatabaseError
from sensai.monitoring.interaction import Interaction, InteractionStatus
from sensai.semantic.cache import RedisSemanticCache
from sensai.tools.tool_manager import ToolManager
from sensai.ui.adapter import CompressionChoice, EventType

if TYPE_CHECKING:
    from collections.abc import Iterator

MODEL = "qwen3:1.7b"
EMBEDDING_MODEL = "nomic-embed-text"


@pytest.fixture
def provider() -> MagicMock:
    provider = MagicMock()
    provider.list.return_value = [MODEL, f"{EMBEDDING_MODEL}:latest"]
    return provider


@pytest.fixture(autouse=True)
def db() -> Iterator[MagicMock]:
    with (
        patch("sensai.core.core.get_db", return_value=MagicMock()) as get_db,
        patch("sensai.core.core.PersonaManager"),
    ):
        yield get_db.return_value


@pytest.fixture
def redis_client() -> Iterator[MagicMock]:
    client = MagicMock()
    with patch("sensai.semantic.connection.redis.Redis.from_url", return_value=client):
        yield client


def make_core(
    provider: MagicMock, embedding_model: str | None = EMBEDDING_MODEL, context: ContextSettings | None = None
) -> Core:
    settings = Settings(
        llm=LLMSettings(embedding_model=embedding_model, providers={"ollama": OllamaSettings()}),
        context=context or ContextSettings(),
    )
    with patch.dict("sensai.llm.provider_manager.PROVIDER_ADAPTERS", {"ollama": lambda _conf: provider}):
        return Core(MagicMock(), settings, Namespace(model=MODEL))


def cache_of(core: Core) -> object:
    return core.conversation._cache  # noqa: SLF001


class TestSemanticCache:
    def test_is_given_to_the_conversation_when_redis_answers(
        self, provider: MagicMock, redis_client: MagicMock
    ) -> None:
        core = make_core(provider)

        redis_client.ping.assert_called_once()
        assert isinstance(cache_of(core), RedisSemanticCache)

    def test_embeds_with_the_configured_embedding_model(self, provider: MagicMock, redis_client: MagicMock) -> None:  # noqa: ARG002
        cache = cache_of(make_core(provider))

        assert isinstance(cache, RedisSemanticCache)
        assert cache._embedder.model == EMBEDDING_MODEL  # noqa: SLF001

    def test_is_disabled_when_redis_is_unreachable(self, provider: MagicMock, redis_client: MagicMock) -> None:
        redis_client.ping.side_effect = redis.ConnectionError

        assert cache_of(make_core(provider)) is None
        redis_client.close.assert_called_once()

    def test_is_disabled_without_embedding_model(self, provider: MagicMock, redis_client: MagicMock) -> None:
        assert cache_of(make_core(provider, embedding_model=None)) is None
        redis_client.ping.assert_not_called()

    def test_unknown_embedding_model_is_refused(self, provider: MagicMock, redis_client: MagicMock) -> None:
        with pytest.raises(ModelError, match="mxbai-embed-large"):
            make_core(provider, embedding_model="mxbai-embed-large")
        redis_client.ping.assert_not_called()


class TestContextBudget:
    def test_is_updated_when_logging_the_interaction_fails(self, provider: MagicMock, db: MagicMock) -> None:
        db.interactions.add.side_effect = DatabaseError("down")
        core = make_core(provider, embedding_model=None)
        interaction = Interaction(
            model=MODEL,
            status=InteractionStatus.OK,
            latency_ms=1.0,
            branch_id=uuid4(),
            metrics=ChatMetrics(prompt_tokens=1000, completion_tokens=50),
        )

        with pytest.raises(DatabaseError):
            core._events.emit(interaction)  # noqa: SLF001

        assert core._ui.update_context_usage.call_args.args[0].used == 1050  # noqa: SLF001


class TestNearTheBudget:
    @staticmethod
    def core_at(provider: MagicMock, used: int) -> tuple[Core, MagicMock]:
        """Build a core whose conversation, on a branch of its own, takes `used` of a budget of 1000 tokens."""
        core = make_core(provider, embedding_model=None, context=ContextSettings(budget=1000, warning_threshold=0.8))
        branch = uuid4()
        core._context_budget.record(  # noqa: SLF001
            Interaction(
                model=MODEL,
                status=InteractionStatus.OK,
                latency_ms=1.0,
                branch_id=branch,
                metrics=ChatMetrics(prompt_tokens=used),
            )
        )
        conversation = MagicMock(uuid=branch)
        core.conversation = conversation
        return core, conversation

    def test_asks_how_to_go_on(self, provider: MagicMock) -> None:
        core, _ = self.core_at(provider, used=900)
        core._ui.ask_compression.return_value = CompressionChoice.COMPRESS  # noqa: SLF001

        core._launch_conv_chat("hello")  # noqa: SLF001

        assert core._ui.ask_compression.call_args.args[0].used == 900  # noqa: SLF001

    def test_compressing_goes_on_with_the_same_conversation(self, provider: MagicMock) -> None:
        core, conversation = self.core_at(provider, used=900)
        core._ui.ask_compression.return_value = CompressionChoice.COMPRESS  # noqa: SLF001

        core._launch_conv_chat("hello")  # noqa: SLF001

        conversation.reset.assert_not_called()
        conversation.chat.assert_called_once_with("hello", core._ui)  # noqa: SLF001

    def test_a_new_conversation_starts_with_the_message(self, provider: MagicMock) -> None:
        core, conversation = self.core_at(provider, used=900)
        core._ui.ask_compression.return_value = CompressionChoice.NEW_CONVERSATION  # noqa: SLF001
        conversation.reset.side_effect = lambda: setattr(conversation, "uuid", None)

        core._launch_conv_chat("hello")  # noqa: SLF001

        conversation.reset.assert_called_once()
        core._ui.load_conversation.assert_called_once_with(conversation.history)  # noqa: SLF001
        conversation.add_conv_to_db.assert_called_once_with(None, "hello")
        conversation.chat.assert_called_once_with("hello", core._ui)  # noqa: SLF001

    def test_is_not_asked_below_the_threshold(self, provider: MagicMock) -> None:
        core, conversation = self.core_at(provider, used=500)

        core._launch_conv_chat("hello")  # noqa: SLF001

        core._ui.ask_compression.assert_not_called()  # noqa: SLF001
        conversation.chat.assert_called_once()


def core_with_mcp_host(
    settings: Settings,
    started: tuple[list[MCPTool], dict[str, MCPStartupError]],
    ui: MagicMock | None = None,
) -> tuple[Core, MagicMock, MagicMock]:
    """Build a Core whose MCP host is a mock reporting `started`, returning it with the host type and tools."""
    provider_manager = MagicMock()
    provider_manager.set_model.return_value = True

    with (
        patch("sensai.core.core.get_db", return_value=MagicMock()),
        patch("sensai.core.core.ProviderManager", return_value=provider_manager),
        patch("sensai.core.core.MCPHost") as mcp_host_type,
        patch("sensai.core.core.ToolManager", wraps=ToolManager) as tool_manager_type,
    ):
        mcp_host_type.return_value.start.return_value = started
        core = Core(ui or MagicMock(), settings, Namespace(model="model"))
    return core, mcp_host_type, tool_manager_type


def test_core_without_mcp_configuration_registers_no_mcp_tool() -> None:
    provider_manager = MagicMock()
    provider_manager.set_model.return_value = True
    built_in_tool = MagicMock()
    ui = MagicMock()
    settings = Settings(features=FeatureSettings(disabled=["tools.example"], confirm=["tools.web_fetch"]))

    with (
        patch("sensai.core.core.get_db", return_value=MagicMock()),
        patch("sensai.core.core.ProviderManager", return_value=provider_manager),
        patch("sensai.core.core.build_tools", return_value=[built_in_tool]) as build_tools,
        patch("sensai.core.core.ToolManager") as tool_manager,
    ):
        Core(ui, settings, Namespace(model="model"))

    build_tools.assert_called_once_with(settings, None, ui.ask_user)
    tool_manager.assert_called_once()
    assert tool_manager.call_args.args[0] == [built_in_tool]
    assert tool_manager.call_args.kwargs == {
        "disabled": {"example"},
        "confirm": {"web_fetch"},
        "approver": ui.ask_confirmation,
    }


def test_core_starts_one_mcp_host_for_every_configured_server() -> None:
    first = MCPServerConfig(name="first", command="first")
    second = MCPServerConfig(name="second", command="second")

    _, mcp_host_type, _ = core_with_mcp_host(Settings(mcp=[first, second]), ([], {}))

    mcp_host_type.assert_called_once_with([first, second])
    mcp_host_type.return_value.start.assert_called_once_with()


def test_core_registers_the_tools_of_the_mcp_host() -> None:
    first_tool = MCPTool(MCPToolInfo("echo", "Echo text", {"type": "object"}), MagicMock(server_name="first"))
    second_tool = MCPTool(MCPToolInfo("echo", "Echo text", {"type": "object"}), MagicMock(server_name="second"))
    settings = Settings(mcp=[MCPServerConfig(name="first", command="first")])

    _, _, tool_manager_type = core_with_mcp_host(settings, ([first_tool, second_tool], {}))

    registered_tools = tool_manager_type.call_args.args[0]
    assert registered_tools[-2:] == [first_tool, second_tool]
    assert ToolManager(registered_tools).names() == ["ask_user", "first__echo", "second__echo"]


def test_core_closes_the_mcp_host_when_the_application_stops() -> None:
    ui = MagicMock()
    ui.get_event.return_value = MagicMock(type=EventType.Command, content="/q")
    settings = Settings(mcp=[MCPServerConfig(name="server", command="server")])
    core, mcp_host_type, _ = core_with_mcp_host(settings, ([], {}), ui)

    core.run()

    mcp_host_type.return_value.close.assert_called_once_with()


def test_core_closes_the_mcp_host_when_the_application_fails() -> None:
    ui = MagicMock()
    ui.open.side_effect = RuntimeError("ui failed")
    settings = Settings(mcp=[MCPServerConfig(name="server", command="server")])
    core, mcp_host_type, _ = core_with_mcp_host(settings, ([], {}), ui)

    with pytest.raises(RuntimeError, match="ui failed"):
        core.run()

    mcp_host_type.return_value.close.assert_called_once_with()


def test_core_reports_mcp_servers_that_failed_to_start() -> None:
    ui = MagicMock()
    ui.get_event.return_value = MagicMock(type=EventType.Command, content="/q")
    settings = Settings(mcp=[MCPServerConfig(name="github", command="server")])
    core, _, _ = core_with_mcp_host(settings, ([], {"github": MCPStartupError("connection refused")}), ui)

    core.run()

    ui.send_system_message.assert_any_call(
        "MCP server 'github' failed to start, its tools are unavailable: connection refused",
    )
