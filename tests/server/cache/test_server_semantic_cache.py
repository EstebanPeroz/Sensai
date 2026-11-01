import contextlib
import os
from collections.abc import AsyncIterator
from typing import override
from uuid import uuid4

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sensai_server.cache import CacheError, SemanticCache, SessionCache
from sensai_server.llm.error import EmbeddingError

from tests.server.fakes import FakeEmbedder

REDIS_URL = os.environ.get("SENSAI_TEST_REDIS_URL", "redis://localhost:6379/0")
"""The Redis the tests run on. Search indexes need db 0, so each test keeps to a namespace of its own."""
UNREACHABLE = "redis://127.0.0.1:1/0"
THRESHOLD = 0.95
TTL = 60

QUESTION = "How does the engine work?"
PARAPHRASE = "how does an engine work"
UNRELATED = "Why do cats sleep so much?"
ANSWER = "It burns fuel."


class WideEmbedder(FakeEmbedder):
    """A `FakeEmbedder` whose vectors have one more dimension, as if the embedding model had changed."""

    @override
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[*vector, 0.0] for vector in super().embed(texts)]


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        await client.ping()
    except RedisError as error:
        await client.aclose()
        pytest.skip(f"no Redis at {REDIS_URL}: {error}")
    yield client
    await client.aclose()


@pytest.fixture
async def namespace(redis: Redis) -> AsyncIterator[str]:
    """A namespace of this test alone, whose index and keys are removed afterwards."""
    name = f"sensai:test:{uuid4().hex}"
    yield name
    with contextlib.suppress(RedisError):  # the test may never have created the index
        await redis.ft(f"{name}:idx").dropindex(delete_documents=True)
    keys = [key async for key in redis.scan_iter(match=f"{name}:*")]
    if keys:
        await redis.unlink(*keys)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
async def cache(namespace: str, embedder: FakeEmbedder) -> AsyncIterator[SemanticCache]:
    built = SemanticCache(REDIS_URL, embedder, THRESHOLD, TTL, namespace=namespace)
    yield built
    await built.close()


async def _keys(redis: Redis, pattern: str) -> list[str]:
    return [key async for key in redis.scan_iter(match=pattern)]


class TestConnect:
    async def test_ping(self, cache: SemanticCache) -> None:
        assert await cache.ping()
        await cache.connect()

    async def test_unreachable(self, embedder: FakeEmbedder) -> None:
        cache = SemanticCache(UNREACHABLE, embedder, THRESHOLD, TTL)
        try:
            assert not await cache.ping()
            with pytest.raises(CacheError, match="does not answer"):
                await cache.connect()
        finally:
            await cache.close()

    async def test_for_session_does_no_io(self, embedder: FakeEmbedder) -> None:
        cache = SemanticCache(UNREACHABLE, embedder, THRESHOLD, TTL)
        try:
            assert isinstance(cache.for_session(uuid4()), SessionCache)
        finally:
            await cache.close()

    async def test_unreachable_lookup_raises(self, embedder: FakeEmbedder) -> None:
        cache = SemanticCache(UNREACHABLE, embedder, THRESHOLD, TTL)
        try:
            with pytest.raises(CacheError):
                await cache.for_session(uuid4()).lookup(QUESTION)
        finally:
            await cache.close()


class TestLookup:
    async def test_empty_cache_misses(self, cache: SemanticCache) -> None:
        assert await cache.for_session(uuid4()).lookup(QUESTION) is None

    async def test_same_prompt_hits(self, cache: SemanticCache) -> None:
        session = cache.for_session(uuid4())
        await session.store(QUESTION, ANSWER)
        assert await session.lookup(QUESTION) == ANSWER

    async def test_near_duplicate_hits(self, cache: SemanticCache) -> None:
        session = cache.for_session(uuid4())
        await session.store(QUESTION, ANSWER)
        assert await session.lookup(PARAPHRASE) == ANSWER

    async def test_unrelated_misses(self, cache: SemanticCache) -> None:
        session = cache.for_session(uuid4())
        await session.store(QUESTION, ANSWER)
        assert await session.lookup(UNRELATED) is None

    async def test_below_threshold_misses(self, namespace: str, embedder: FakeEmbedder) -> None:
        strict = SemanticCache(REDIS_URL, embedder, 1.01, TTL, namespace=namespace)
        try:
            session = strict.for_session(uuid4())
            await session.store(QUESTION, ANSWER)
            assert await session.lookup(QUESTION) is None
        finally:
            await strict.close()

    async def test_embedding_error_propagates(self, cache: SemanticCache, embedder: FakeEmbedder) -> None:
        embedder.failing = True
        with pytest.raises(EmbeddingError):
            await cache.for_session(uuid4()).lookup(QUESTION)

    async def test_most_similar_answer_wins(self, cache: SemanticCache) -> None:
        session = cache.for_session(uuid4())
        await session.store(UNRELATED, "about cats")
        await session.store(QUESTION, ANSWER)
        assert await session.lookup(PARAPHRASE) == ANSWER
        assert await session.lookup(UNRELATED) == "about cats"

    def test_threshold_comes_from_the_settings(self, cache: SemanticCache) -> None:
        assert cache.similarity_threshold == THRESHOLD

    async def test_unicode_round_trips(self, cache: SemanticCache) -> None:
        session = cache.for_session(uuid4())
        answer = "Le moteur brûle du carburant 🔥 — 燃料"
        await session.store(QUESTION, answer)
        assert await session.lookup(QUESTION) == answer


class TestStore:
    async def test_reuses_lookup_vector(self, cache: SemanticCache, embedder: FakeEmbedder) -> None:
        session = cache.for_session(uuid4())
        await session.lookup(QUESTION)
        await session.store(QUESTION, ANSWER)
        assert embedder.batches == [[QUESTION]]

    async def test_embeds_another_prompt(self, cache: SemanticCache, embedder: FakeEmbedder) -> None:
        session = cache.for_session(uuid4())
        await session.lookup(QUESTION)
        await session.store(UNRELATED, ANSWER)
        assert embedder.batches == [[QUESTION], [UNRELATED]]

    async def test_embedding_error_propagates_and_stores_nothing(
        self, cache: SemanticCache, embedder: FakeEmbedder, redis: Redis, namespace: str
    ) -> None:
        embedder.failing = True
        with pytest.raises(EmbeddingError):
            await cache.for_session(uuid4()).store(QUESTION, ANSWER)
        assert await _keys(redis, f"{namespace}:*") == []

    async def test_unreachable_store_raises(self, embedder: FakeEmbedder) -> None:
        cache = SemanticCache(UNREACHABLE, embedder, THRESHOLD, TTL)
        try:
            with pytest.raises(CacheError):
                await cache.for_session(uuid4()).store(QUESTION, ANSWER)
        finally:
            await cache.close()

    async def test_entry_layout(self, cache: SemanticCache, redis: Redis, namespace: str) -> None:
        client_id = uuid4()
        await cache.for_session(client_id).store(QUESTION, ANSWER)
        [key] = await _keys(redis, f"{namespace}:*")
        assert key.startswith(f"{namespace}:{client_id}:")
        assert await redis.hget(key, "session") == client_id.hex
        assert await redis.hget(key, "prompt") == QUESTION
        assert await redis.hget(key, "answer") == ANSWER
        assert 0 < await redis.ttl(key) <= TTL

    async def test_dimension_change_recreates_index(self, cache: SemanticCache, namespace: str) -> None:
        session = cache.for_session(uuid4())
        await session.store(QUESTION, ANSWER)
        wide = SemanticCache(REDIS_URL, WideEmbedder(), THRESHOLD, TTL, namespace=namespace)
        try:
            wide_session = wide.for_session(uuid4())
            assert await wide_session.lookup(QUESTION) is None
            await wide_session.store(QUESTION, "wide")
            assert await wide_session.lookup(QUESTION) == "wide"
        finally:
            await wide.close()


class TestIsolation:
    async def test_sessions_never_see_each_other(self, cache: SemanticCache) -> None:
        first = cache.for_session(uuid4())
        second = cache.for_session(uuid4())
        await first.store(QUESTION, "first")
        assert await second.lookup(QUESTION) is None
        await second.store(QUESTION, "second")
        assert await first.lookup(QUESTION) == "first"
        assert await second.lookup(QUESTION) == "second"

    async def test_clear_only_removes_own_entries(self, cache: SemanticCache, redis: Redis, namespace: str) -> None:
        first_id, second_id = uuid4(), uuid4()
        first = cache.for_session(first_id)
        second = cache.for_session(second_id)
        await first.store(QUESTION, "first")
        await second.store(QUESTION, "second")
        await first.clear()
        assert await _keys(redis, f"{namespace}:{first_id}:*") == []
        assert len(await _keys(redis, f"{namespace}:{second_id}:*")) == 1
        assert await first.lookup(QUESTION) is None
        assert await second.lookup(QUESTION) == "second"

    async def test_clear_all(self, cache: SemanticCache, redis: Redis, namespace: str) -> None:
        await cache.for_session(uuid4()).store(QUESTION, "first")
        await cache.for_session(uuid4()).store(QUESTION, "second")
        await cache.clear_all()
        assert await _keys(redis, f"{namespace}:*") == []

    async def test_clear_all_keeps_other_namespaces(self, cache: SemanticCache, redis: Redis, namespace: str) -> None:
        other = f"{namespace}-other:key"
        await redis.set(other, "kept")
        try:
            await cache.clear_all()
            assert await redis.get(other) == "kept"
        finally:
            await redis.unlink(other)
