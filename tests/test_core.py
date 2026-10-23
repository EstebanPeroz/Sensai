from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import redis

from sensai.config.settings import LLMSettings, OllamaSettings, Settings
from sensai.core.core import Core
from sensai.llm.error import ModelError
from sensai.semantic.cache import RedisSemanticCache

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
    with patch("sensai.core.core.redis.Redis.from_url", return_value=client):
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
