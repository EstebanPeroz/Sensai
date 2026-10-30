from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
import redis

from sensai.config.settings import FeatureSettings, LLMSettings, OllamaSettings, Settings
from sensai.core.core import Core
from sensai.llm.error import ModelError
from sensai.llm.responses import ChatMetrics
from sensai.mcp.config import MCPServerConfig
from sensai.mcp.discovery import MCPToolInfo
from sensai.mcp.tool import MCPTool
from sensai.memory.persistent_db.error import DatabaseError
from sensai.monitoring.interaction import Interaction, InteractionStatus
from sensai.semantic.cache import RedisSemanticCache
from sensai.tools.tool_manager import ToolManager
from sensai.ui.adapter import EventType

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


def make_core(provider: MagicMock, embedding_model: str | None = EMBEDDING_MODEL) -> Core:
    settings = Settings(llm=LLMSettings(embedding_model=embedding_model, providers={"ollama": OllamaSettings()}))
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


def test_core_does_not_create_mcp_client_without_configuration() -> None:
    provider_manager = MagicMock()
    provider_manager.set_model.return_value = True
    db = MagicMock()
    built_in_tool = MagicMock()
    ui = MagicMock()
    settings = Settings(features=FeatureSettings(disabled=["tools.example"], confirm=["tools.web_fetch"]))

    with (
        patch("sensai.core.core.get_db", return_value=db),
        patch(
            "sensai.core.core.ProviderManager",
            return_value=provider_manager,
        ),
        patch(
            "sensai.core.core.MCPClient",
        ) as mcp_client,
        patch(
            "sensai.core.core.build_tools",
            return_value=[built_in_tool],
        ) as build_tools,
        patch(
            "sensai.core.core.ToolManager",
        ) as tool_manager,
    ):
        Core(ui, settings, Namespace(model="model"))

    mcp_client.assert_not_called()
    build_tools.assert_called_once_with(settings, None, ui.ask_user)
    tool_manager.assert_called_once()
    assert tool_manager.call_args.args[0] == [built_in_tool]
    assert tool_manager.call_args.kwargs == {
        "disabled": {"example"},
        "confirm": {"web_fetch"},
        "approver": ui.ask_confirmation,
    }


def test_core_creates_mcp_client_for_configured_server() -> None:
    provider_manager = MagicMock()
    provider_manager.set_model.return_value = True
    config = MCPServerConfig(name="server", command="server")
    db = MagicMock()

    with (
        patch("sensai.core.core.get_db", return_value=db),
        patch(
            "sensai.core.core.ProviderManager",
            return_value=provider_manager,
        ),
        patch(
            "sensai.core.core.MCPClient",
        ) as mcp_client,
        patch(
            "sensai.core.core.anyio.run",
            return_value=[],
        ),
    ):
        Core(MagicMock(), Settings(mcp=[config]), Namespace(model="model"))

    mcp_client.assert_called_once_with(config)


def test_core_creates_one_mcp_client_per_configured_server() -> None:
    provider_manager = MagicMock()
    provider_manager.set_model.return_value = True
    first = MCPServerConfig(name="first", command="first")
    second = MCPServerConfig(name="second", command="second")
    first_tool = MCPTool(MCPToolInfo("echo", "Echo text", {"type": "object"}), MagicMock())
    second_tool = MCPTool(MCPToolInfo("search", "Search text", {"type": "object"}), MagicMock())

    with (
        patch("sensai.core.core.get_db", return_value=MagicMock()),
        patch(
            "sensai.core.core.ProviderManager",
            return_value=provider_manager,
        ),
        patch(
            "sensai.core.core.MCPClient",
        ) as mcp_client_type,
        patch(
            "sensai.core.core.anyio.run",
            side_effect=[[first_tool], [second_tool]],
        ),
        patch(
            "sensai.core.core.ToolManager",
        ) as tool_manager_type,
    ):
        Core(MagicMock(), Settings(mcp=[first, second]), Namespace(model="model"))

    assert [call.args for call in mcp_client_type.call_args_list] == [(first,), (second,)]
    assert tool_manager_type.call_args.args[0][-2:] == [first_tool, second_tool]


def test_core_registers_discovered_mcp_tools() -> None:
    provider_manager = MagicMock()
    provider_manager.set_model.return_value = True
    db = MagicMock()
    config = MCPServerConfig(name="server", command="server")
    mcp_tool = MCPTool(MCPToolInfo("echo", "Echo text", {"type": "object"}), MagicMock())

    with (
        patch("sensai.core.core.get_db", return_value=db),
        patch(
            "sensai.core.core.ProviderManager",
            return_value=provider_manager,
        ),
        patch(
            "sensai.core.core.MCPClient",
        ) as mcp_client_type,
        patch(
            "sensai.core.core.anyio.run",
            return_value=[mcp_tool],
        ) as run,
        patch(
            "sensai.core.core.ToolManager",
        ) as tool_manager_type,
    ):
        Core(MagicMock(), Settings(mcp=[config]), Namespace(model="model"))

    run.assert_called_once_with(mcp_client_type.return_value.start)
    tool_manager_type.assert_called_once()
    registered_tools = tool_manager_type.call_args.args[0]
    assert registered_tools[-1] is mcp_tool
    assert ToolManager(registered_tools).names() == ["ask_user", "echo"]


def test_core_runs_with_mcp_client() -> None:
    provider_manager = MagicMock()
    provider_manager.set_model.return_value = True
    db = MagicMock()
    ui = MagicMock()
    ui.get_event.return_value = MagicMock(type=EventType.Command, content="/q")

    with (
        patch("sensai.core.core.get_db", return_value=db),
        patch(
            "sensai.core.core.ProviderManager",
            return_value=provider_manager,
        ),
        patch(
            "sensai.core.core.MCPClient",
        ),
        patch(
            "sensai.core.core.anyio.run",
            return_value=[],
        ) as run,
    ):
        core = Core(ui, Settings(mcp=[MCPServerConfig(name="server", command="server")]), Namespace(model="model"))
        core.run()

    run.assert_called_once()
