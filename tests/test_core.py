from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import redis

from sensai.config.settings import LLMSettings, OllamaSettings, Settings
from sensai.core.core import Core
from sensai.llm.provider_manager import ProviderManager
from sensai.memory.semantic_cache import RedisSemanticCache

if TYPE_CHECKING:
    from collections.abc import Iterator

EMBEDDING_MODEL = "nomic-embed-text"


@pytest.fixture
def provider() -> MagicMock:
    provider = MagicMock()
    provider.list.return_value = ["qwen3:1.7b", f"{EMBEDDING_MODEL}:latest"]
    return provider


@pytest.fixture
def redis_client() -> Iterator[MagicMock]:
    client = MagicMock()
    with patch("sensai.core.core.redis.Redis.from_url", return_value=client):
        yield client


def make_core(provider: MagicMock, embedding_model: str | None = EMBEDDING_MODEL) -> Core:
    settings = Settings(llm=LLMSettings(embedding_model=embedding_model, providers={"ollama": OllamaSettings()}))
    with patch.dict("sensai.llm.provider_manager.PROVIDER_ADAPTERS", {"ollama": lambda _conf: provider}):
        return Core(MagicMock(), settings)


class TestSemanticCache:
    def test_is_built_when_redis_answers(self, provider: MagicMock, redis_client: MagicMock) -> None:
        core = make_core(provider)

        redis_client.ping.assert_called_once()
        assert isinstance(core._cache, RedisSemanticCache)  # noqa: SLF001

    def test_is_given_to_the_conversation(self, provider: MagicMock, redis_client: MagicMock) -> None:  # noqa: ARG002
        core = make_core(provider)

        core.init_conversation("qwen3:1.7b")

        assert core.conversation is not None
        assert core.conversation._cache is core._cache  # noqa: SLF001

    def test_is_disabled_when_redis_is_unreachable(self, provider: MagicMock, redis_client: MagicMock) -> None:
        redis_client.ping.side_effect = redis.ConnectionError

        assert make_core(provider)._cache is None  # noqa: SLF001

    def test_is_disabled_without_embedding_model(self, provider: MagicMock, redis_client: MagicMock) -> None:
        assert make_core(provider, embedding_model=None)._cache is None  # noqa: SLF001
        redis_client.ping.assert_not_called()

    def test_is_disabled_when_no_provider_serves_the_embedding_model(
        self, provider: MagicMock, redis_client: MagicMock
    ) -> None:
        assert make_core(provider, embedding_model="mxbai-embed-large")._cache is None  # noqa: SLF001
        redis_client.ping.assert_not_called()


class TestProviderManager:
    def test_untagged_model_resolves_to_latest(self, provider: MagicMock) -> None:
        with patch.dict("sensai.llm.provider_manager.PROVIDER_ADAPTERS", {"ollama": lambda _conf: provider}):
            manager = ProviderManager(LLMSettings(providers={"ollama": OllamaSettings()}))

        assert manager.get_provider(EMBEDDING_MODEL) is provider
        assert manager.get_provider("qwen3:1.7b") is provider
        assert manager.get_provider("qwen3") is None
