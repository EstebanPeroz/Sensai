import functools
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from redis.asyncio import Redis
from sensai_server.app import app as app_module
from sensai_server.app import create_app
from sensai_server.app._state import app_state
from sensai_server.cache import SemanticCache
from sensai_server.config.settings import CacheSettings, Settings
from sensai_server.pipeline import ChatPipeline
from sensai_server.pipeline.modules import CacheStage
from sensai_server.pipeline.modules.guardrails import BadWordsGuard

from tests.server.conftest import MODEL, settings_with_db
from tests.server.fakes import FakeEmbedder
from tests.server.test_server_semantic_cache import REDIS_URL, UNREACHABLE


def _stage_types(pipeline: ChatPipeline) -> list[type]:
    return [type(stage) for stage in pipeline._stages]  # noqa: SLF001


def _settings(tmp_path: Path, redis_url: str) -> Settings:
    base = settings_with_db(tmp_path)
    return Settings(llm=base.llm, db=base.db, cache=CacheSettings(redis_url=redis_url))


@pytest.fixture
def embedder(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    """Give `create_app` an embedder, as if the settings named an embedding model served by a provider."""
    built = FakeEmbedder()
    monkeypatch.setattr(app_module, "_build_embedder", lambda _settings, _pm: built)
    return built


class TestWithoutCache:
    def test_no_embedder_leaves_the_cache_out(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            app = create_app(settings_with_db(tmp_path))

        assert app_state(app).cache is None
        assert CacheStage not in _stage_types(app_state(app).pipeline)
        assert "no embedder: the semantic cache is left out" in caplog.text

    @pytest.mark.usefixtures("embedder")
    def test_unreachable_redis_leaves_the_cache_out(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            app = create_app(_settings(tmp_path, UNREACHABLE))

        assert app_state(app).cache is None
        assert CacheStage not in _stage_types(app_state(app).pipeline)
        assert "the Redis server does not answer" in caplog.text

    def test_the_app_starts_and_stops(self, tmp_path: Path) -> None:
        with TestClient(create_app(settings_with_db(tmp_path))) as client, client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "login", "model": MODEL})
            assert ws.receive_json()["type"] == "session"


class TestWithCache:
    @pytest.fixture
    async def redis(self) -> AsyncIterator[Redis]:
        client = Redis.from_url(REDIS_URL, decode_responses=True)
        if not app_module._redis_answers(REDIS_URL):  # noqa: SLF001
            await client.aclose()
            pytest.skip(f"no Redis at {REDIS_URL}")
        yield client
        await client.aclose()

    @pytest.fixture
    async def namespace(self, redis: Redis, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[str]:
        """Have `create_app` build its cache under a namespace of this test alone, removed afterwards."""
        name = f"sensai:test:{uuid4().hex}"
        monkeypatch.setattr(app_module, "SemanticCache", functools.partial(SemanticCache, namespace=name))
        yield name
        keys = [key async for key in redis.scan_iter(match=f"{name}:*")]
        if keys:
            await redis.unlink(*keys)

    @pytest.mark.usefixtures("embedder")
    async def test_builds_the_cache(self, tmp_path: Path, namespace: str) -> None:  # noqa: ARG002
        app = create_app(_settings(tmp_path, REDIS_URL))

        assert isinstance(app_state(app).cache, SemanticCache)
        assert _stage_types(app_state(app).pipeline)[:2] == [BadWordsGuard, CacheStage]

    @pytest.mark.usefixtures("embedder")
    async def test_startup_clears_the_entries_of_a_previous_run(
        self, tmp_path: Path, redis: Redis, namespace: str
    ) -> None:
        stale = f"{namespace}:{uuid4()}:{uuid4()}"
        await redis.hset(stale, mapping={"prompt": "p", "answer": "a"})
        app = create_app(_settings(tmp_path, REDIS_URL))

        with TestClient(app):
            assert await redis.exists(stale) == 0

    @pytest.mark.usefixtures("embedder")
    async def test_shutdown_closes_the_sessions_and_leaves_no_entry(
        self, tmp_path: Path, redis: Redis, namespace: str
    ) -> None:
        app = create_app(_settings(tmp_path, REDIS_URL))
        state = app_state(app)
        with TestClient(app) as client, client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "login", "model": MODEL})
            assert ws.receive_json()["type"] == "session"
            assert state.cache is not None
            await redis.hset(f"{namespace}:{uuid4()}:{uuid4()}", mapping={"prompt": "p", "answer": "a"})

        assert len(state.registry) == 0
        assert [key async for key in redis.scan_iter(match=f"{namespace}:*")] == []
