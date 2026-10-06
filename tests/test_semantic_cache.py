from __future__ import annotations

import fnmatch
from typing import Self

import pytest
import redis

from sensai.config.settings import CacheSettings
from sensai.error import SensaiError
from sensai.memory.semantic_cache import KEY_PREFIX, RedisSemanticCache
from sensai.memory.semantic_cache_repository import CacheScope, SemanticCacheRepository

SCOPE = CacheScope(model="llama3", persona="analyst")

EMBEDDINGS = {
    "what is redis?": [1.0, 0.0, 0.0],
    "what's redis?": [0.99, 0.1, 0.0],
    "explain redis": [0.8, 0.6, 0.0],
    "how to cook pasta?": [0.0, 0.0, 1.0],
    "give me an example": [0.0, 1.0, 0.0],
    "show me an example": [0.05, 0.99, 0.0],
}


def embed(text: str, model: str) -> list[float]:
    if text == "embedding failure":
        raise SensaiError(model)
    return EMBEDDINGS.get(text, [])


class FakeRedis:
    """In-memory stand-in for the few redis.Redis calls the cache makes."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, bytes]] = {}
        self.ttls: dict[str, int] = {}
        self.fail = False

    def _check(self) -> None:
        if self.fail:
            raise redis.ConnectionError

    def scan_iter(self, match: str) -> list[str]:
        self._check()
        return [key for key in self.hashes if fnmatch.fnmatchcase(key, match)]

    def hmget(self, key: str, fields: list[str]) -> list[bytes | None]:
        self._check()
        entry = self.hashes.get(key, {})
        return [entry.get(name) for name in fields]

    def delete(self, *keys: str) -> None:
        self._check()
        for key in keys:
            self.hashes.pop(key, None)

    def pipeline(self) -> Self:
        return self

    def hset(self, key: str, mapping: dict[str, str | bytes]) -> None:
        self.hashes[key] = {
            name: value.encode() if isinstance(value, str) else value for name, value in mapping.items()
        }

    def expire(self, key: str, seconds: int) -> None:
        self.ttls[key] = seconds

    def execute(self) -> None:
        self._check()


@pytest.fixture
def client() -> FakeRedis:
    return FakeRedis()


def make_cache(
    client: FakeRedis, embedding_model: str = "nomic-embed-text", **settings: float
) -> SemanticCacheRepository:
    return RedisSemanticCache(client, CacheSettings(**settings), embed, embedding_model)  # type: ignore[arg-type]


class TestLookup:
    def test_empty_cache_misses(self, client: FakeRedis) -> None:
        assert make_cache(client).lookup("what is redis?", SCOPE) is None

    def test_same_query_hits(self, client: FakeRedis) -> None:
        cache = make_cache(client)
        cache.store("what is redis?", SCOPE, "An in-memory store.")

        assert cache.lookup("what is redis?", SCOPE) == "An in-memory store."

    def test_similar_query_hits(self, client: FakeRedis) -> None:
        cache = make_cache(client)
        cache.store("what is redis?", SCOPE, "An in-memory store.")

        assert cache.lookup("what's redis?", SCOPE) == "An in-memory store."

    def test_query_below_threshold_misses(self, client: FakeRedis) -> None:
        cache = make_cache(client)
        cache.store("what is redis?", SCOPE, "An in-memory store.")

        assert cache.lookup("explain redis", SCOPE) is None
        assert cache.lookup("how to cook pasta?", SCOPE) is None

    def test_threshold_comes_from_settings(self, client: FakeRedis) -> None:
        cache = make_cache(client, similarity_threshold=0.5)
        cache.store("what is redis?", SCOPE, "An in-memory store.")

        assert cache.lookup("explain redis", SCOPE) == "An in-memory store."

    def test_most_similar_answer_wins(self, client: FakeRedis) -> None:
        cache = make_cache(client, similarity_threshold=0.5)
        cache.store("explain redis", SCOPE, "far")
        cache.store("what's redis?", SCOPE, "close")

        assert cache.lookup("what is redis?", SCOPE) == "close"

    @pytest.mark.parametrize(
        "other",
        [
            CacheScope(model="mistral", persona="analyst"),
            CacheScope(model="llama3", persona="teacher"),
            CacheScope(model="llama3", persona="analyst", subject="databases"),
        ],
    )
    def test_answer_is_not_served_out_of_its_scope(self, client: FakeRedis, other: CacheScope) -> None:
        cache = make_cache(client)
        cache.store("what is redis?", SCOPE, "An in-memory store.")

        assert cache.lookup("what is redis?", other) is None

    def test_answer_is_not_served_to_another_embedding_model(self, client: FakeRedis) -> None:
        make_cache(client).store("what is redis?", SCOPE, "An in-memory store.")

        assert make_cache(client, embedding_model="mxbai-embed-large").lookup("what is redis?", SCOPE) is None

    def test_query_without_embedding_misses(self, client: FakeRedis) -> None:
        cache = make_cache(client)
        cache.store("what is redis?", SCOPE, "An in-memory store.")

        assert cache.lookup("unknown", SCOPE) is None

    def test_redis_failure_is_a_miss(self, client: FakeRedis) -> None:
        cache = make_cache(client)
        cache.store("what is redis?", SCOPE, "An in-memory store.")
        client.fail = True

        assert cache.lookup("what is redis?", SCOPE) is None

    def test_embedding_failure_is_a_miss(self, client: FakeRedis) -> None:
        assert make_cache(client).lookup("embedding failure", SCOPE) is None

    def test_query_is_embedded_with_the_embedding_model(self, client: FakeRedis) -> None:
        models: list[str] = []
        cache = RedisSemanticCache(
            client,  # type: ignore[arg-type]
            CacheSettings(),
            lambda text, model: models.append(model) or embed(text, model),
            "mxbai-embed-large",
        )

        cache.lookup("what is redis?", SCOPE)

        assert models == ["mxbai-embed-large"]


class TestContext:
    def test_same_query_after_a_similar_context_hits(self, client: FakeRedis) -> None:
        cache = make_cache(client)
        cache.store("give me an example", SCOPE, "SET key value", context="what is redis?")

        assert cache.lookup("show me an example", SCOPE, context="what's redis?") == "SET key value"

    def test_same_query_after_another_context_misses(self, client: FakeRedis) -> None:
        cache = make_cache(client)
        cache.store("give me an example", SCOPE, "SET key value", context="what is redis?")

        assert cache.lookup("give me an example", SCOPE, context="how to cook pasta?") is None

    def test_other_query_after_the_same_context_misses(self, client: FakeRedis) -> None:
        cache = make_cache(client)
        cache.store("give me an example", SCOPE, "SET key value", context="what is redis?")

        assert cache.lookup("how to cook pasta?", SCOPE, context="what is redis?") is None

    def test_query_without_context_misses_an_entry_with_context(self, client: FakeRedis) -> None:
        cache = make_cache(client)
        cache.store("give me an example", SCOPE, "SET key value", context="what is redis?")

        assert cache.lookup("give me an example", SCOPE) is None

    def test_query_with_context_misses_an_entry_without_context(self, client: FakeRedis) -> None:
        cache = make_cache(client)
        cache.store("what is redis?", SCOPE, "An in-memory store.")

        assert cache.lookup("what is redis?", SCOPE, context="how to cook pasta?") is None

    def test_context_without_embedding_is_not_stored(self, client: FakeRedis) -> None:
        make_cache(client).store("what is redis?", SCOPE, "An in-memory store.", context="unknown")

        assert client.hashes == {}


class TestStore:
    def test_entry_expires_after_ttl(self, client: FakeRedis) -> None:
        make_cache(client, ttl=60).store("what is redis?", SCOPE, "An in-memory store.")

        assert list(client.ttls.values()) == [60]

    def test_entries_are_namespaced(self, client: FakeRedis) -> None:
        make_cache(client).store("what is redis?", SCOPE, "An in-memory store.")

        assert all(key.startswith(KEY_PREFIX) for key in client.hashes)

    def test_query_without_embedding_is_not_stored(self, client: FakeRedis) -> None:
        make_cache(client).store("unknown", SCOPE, "answer")

        assert client.hashes == {}

    def test_redis_failure_is_ignored(self, client: FakeRedis) -> None:
        client.fail = True

        make_cache(client).store("what is redis?", SCOPE, "An in-memory store.")

    def test_embedding_failure_is_not_stored(self, client: FakeRedis) -> None:
        make_cache(client).store("embedding failure", SCOPE, "answer")

        assert client.hashes == {}


class TestClear:
    def test_removes_every_scope(self, client: FakeRedis) -> None:
        cache = make_cache(client)
        other = CacheScope(model="mistral", persona="analyst")
        cache.store("what is redis?", SCOPE, "a")
        cache.store("what is redis?", other, "b")

        cache.clear()

        assert cache.lookup("what is redis?", SCOPE) is None
        assert cache.lookup("what is redis?", other) is None

    def test_removes_every_embedding_model(self, client: FakeRedis) -> None:
        make_cache(client).store("what is redis?", SCOPE, "a")
        make_cache(client, embedding_model="mxbai-embed-large").store("what is redis?", SCOPE, "b")

        make_cache(client).clear()

        assert client.hashes == {}

    def test_keeps_keys_outside_the_cache(self, client: FakeRedis) -> None:
        client.hashes["other:key"] = {}

        make_cache(client).clear()

        assert list(client.hashes) == ["other:key"]

    def test_redis_failure_is_ignored(self, client: FakeRedis) -> None:
        client.fail = True

        make_cache(client).clear()
