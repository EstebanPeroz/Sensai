import asyncio
import logging
from collections.abc import AsyncIterator
from typing import cast
from uuid import uuid4

import pytest
from redis.asyncio import Redis
from sensai_server.cache import SemanticCache, SessionCache
from sensai_server.llm.persona import PersonaManager
from sensai_server.llm.provider import ProviderManager
from sensai_server.session import Session, SessionRegistry
from sensai_server.tools.tool_manager import ToolManager

from tests.server.conftest import MODEL, SETTINGS
from tests.server.fakes import FakeEmbedder, FakeSessionCache, RecordingOutbound
from tests.server.test_server_semantic_cache import ANSWER, QUESTION, REDIS_URL, THRESHOLD, TTL, UNREACHABLE

# The Redis fixtures of the semantic cache tests: a client, skipping without Redis, and a namespace of the test alone.
from tests.server.test_server_semantic_cache import namespace, redis  # noqa: F401  # isort: skip


@pytest.fixture
async def cache(namespace: str) -> AsyncIterator[SemanticCache]:  # noqa: F811
    built = SemanticCache(REDIS_URL, FakeEmbedder(), THRESHOLD, TTL, namespace=namespace)
    yield built
    await built.close()


def _registry(persona_manager: PersonaManager, cache: SemanticCache | None) -> SessionRegistry:
    return SessionRegistry(ProviderManager(SETTINGS.llm), persona_manager, ToolManager(), cache)


async def _keys(redis: Redis, namespace: str, session: Session) -> list[str]:  # noqa: F811
    return [key async for key in redis.scan_iter(match=f"{namespace}:{session.client_id}:*")]


async def _open(registry: SessionRegistry) -> Session:
    session = await registry.open(RecordingOutbound(), MODEL)
    assert session is not None
    return session


class TestSessionCache:
    async def test_no_cache_gives_no_session_cache(self, persona_manager: PersonaManager) -> None:
        session = await _open(_registry(persona_manager, None))
        assert session.cache is None
        await session.close()

    async def test_each_session_gets_its_own_view(self, persona_manager: PersonaManager, cache: SemanticCache) -> None:
        registry = _registry(persona_manager, cache)
        first, second = await _open(registry), await _open(registry)
        assert first.cache is not None
        assert second.cache is not None
        assert first.cache is not second.cache
        await first.cache.store(QUESTION, ANSWER)
        assert await first.cache.lookup(QUESTION) == ANSWER
        assert await second.cache.lookup(QUESTION) is None


class TestCloseClearsTheCache:
    async def test_close_clears_only_its_own_entries(
        self,
        persona_manager: PersonaManager,
        cache: SemanticCache,
        redis: Redis,  # noqa: F811
        namespace: str,  # noqa: F811
    ) -> None:
        registry = _registry(persona_manager, cache)
        leaving, staying = await _open(registry), await _open(registry)
        assert leaving.cache is not None
        assert staying.cache is not None
        await leaving.cache.store(QUESTION, ANSWER)
        await staying.cache.store(QUESTION, ANSWER)

        await registry.close(leaving.client_id)

        assert await _keys(redis, namespace, leaving) == []
        assert len(await _keys(redis, namespace, staying)) == 1
        assert await staying.cache.lookup(QUESTION) == ANSWER

    async def test_close_all_clears_every_session(
        self,
        persona_manager: PersonaManager,
        cache: SemanticCache,
        redis: Redis,  # noqa: F811
        namespace: str,  # noqa: F811
    ) -> None:
        registry = _registry(persona_manager, cache)
        sessions = [await _open(registry), await _open(registry)]
        for session in sessions:
            assert session.cache is not None
            await session.cache.store(QUESTION, ANSWER)

        await registry.close_all()

        assert [key async for key in redis.scan_iter(match=f"{namespace}:*")] == []

    async def test_close_stops_the_task_before_clearing(
        self,
        persona_manager: PersonaManager,
        cache: SemanticCache,
        redis: Redis,  # noqa: F811
        namespace: str,  # noqa: F811
    ) -> None:
        registry = _registry(persona_manager, cache)
        session = await _open(registry)
        assert session.cache is not None
        session_cache = session.cache
        started = asyncio.Event()

        async def store_when_cancelled() -> None:
            started.set()
            try:
                await asyncio.sleep(3600)
            finally:
                await session_cache.store(QUESTION, ANSWER)

        session.start(store_when_cancelled())
        await started.wait()
        await registry.close(session.client_id)

        assert await _keys(redis, namespace, session) == []

    async def test_a_failing_clear_does_not_stop_the_close(
        self, persona_manager: PersonaManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        unreachable = SemanticCache(UNREACHABLE, FakeEmbedder(), THRESHOLD, TTL, namespace=f"sensai:test:{uuid4().hex}")
        try:
            registry = _registry(persona_manager, unreachable)
            session = await _open(registry)
            started = asyncio.Event()

            async def forever() -> None:
                started.set()
                await asyncio.sleep(3600)

            session.start(forever())
            await started.wait()
            task = session.task
            with caplog.at_level(logging.WARNING):
                await registry.close(session.client_id)

            assert len(registry) == 0
            assert task is not None
            assert task.cancelled()
            assert "could not be cleared" in caplog.text
        finally:
            await unreachable.close()


class TestCloseWithAFakeCache:
    """The close rules, checked without Redis."""

    async def test_close_clears_the_cache(self, persona_manager: PersonaManager) -> None:
        registry = _registry(persona_manager, None)
        session = await _open(registry)
        fake = FakeSessionCache({QUESTION: ANSWER})
        session.cache = cast("SessionCache", fake)

        await registry.close(session.client_id)

        assert fake.clears == 1
        assert fake.answers == {}

    async def test_a_failing_clear_does_not_stop_the_close(
        self, persona_manager: PersonaManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        registry = _registry(persona_manager, None)
        session = await _open(registry)
        session.cache = cast("SessionCache", FakeSessionCache(failing_clear=True))

        with caplog.at_level(logging.WARNING):
            await registry.close(session.client_id)

        assert len(registry) == 0
        assert "could not be cleared" in caplog.text
