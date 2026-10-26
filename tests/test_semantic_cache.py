from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING

import pytest
import redis

from sensai.config.settings import CacheSettings
from sensai.error import SensaiError
from sensai.llm.embedder import Embedder
from sensai.semantic.cache import KEY_PREFIX, RedisSemanticCache
from sensai.semantic.cache_repository import CacheScope, SemanticCacheRepository

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

SCOPE = CacheScope(model="llama3", persona="analyst")

EMBEDDINGS = {
    "what is redis?": [1.0, 0.0, 0.0],
    "what's redis?": [0.99, 0.1, 0.0],
    "explain redis": [0.8, 0.6, 0.0],
    "how to cook pasta?": [0.0, 0.0, 1.0],
    "give me an example": [0.0, 1.0, 0.0],
    "show me an example": [0.05, 0.99, 0.0],
}


class FakeProvider:
    """Embeds the texts of EMBEDDINGS, fails on "embedding failure" and returns nothing for any other text."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def embeddings(self, messages: list[str], model: str) -> list[list[float]]:
        self.calls.append((messages, model))
        if "embedding failure" in messages:
            raise SensaiError(model)
        if not all(text in EMBEDDINGS for text in messages):
            return []
        return [EMBEDDINGS[text] for text in messages]


class FakeRedis:
    """In-memory stand-in for the few redis.Redis calls the cache makes."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, bytes]] = {}
        self.ttls: dict[str, int] = {}
        self.fail = False

    def _check(self) -> None:
        if self.fail:
            raise redis.ConnectionError

    def scan_iter(self, match: str, count: int | None = None) -> list[str]:  # noqa: ARG002
        self._check()
        return [key for key in self.hashes if fnmatch.fnmatchcase(key, match)]

    def hmget(self, key: str, fields: list[str]) -> list[bytes | None]:
        self._check()
        entry = self.hashes.get(key, {})
        return [entry.get(name) for name in fields]

    def pipeline(self, *, transaction: bool = True) -> FakePipeline:  # noqa: ARG002
        return FakePipeline(self)


class FakePipeline:
    """Queues the commands sent to a FakeRedis, and runs them on execute."""

    def __init__(self, client: FakeRedis) -> None:
        self._client = client
        self._commands: list[Callable[[], object]] = []

    def hmget(self, key: str, fields: Sequence[str]) -> None:
        self._commands.append(lambda: self._client.hmget(key, list(fields)))

    def hset(self, key: str, mapping: dict[str, str | bytes]) -> None:
        def run() -> None:
            self._client.hashes[key] = {
                name: value.encode() if isinstance(value, str) else value for name, value in mapping.items()
            }

        self._commands.append(run)

    def expire(self, key: str, seconds: int) -> None:
        self._commands.append(lambda: self._client.ttls.__setitem__(key, seconds))

    def execute(self) -> list[object]:
        self._client._check()  # noqa: SLF001
        return [command() for command in self._commands]


@pytest.fixture
def client() -> FakeRedis:
    return FakeRedis()


def make_cache(
    client: FakeRedis,
    embedding_model: str = "nomic-embed-text",
    provider: FakeProvider | None = None,
    **settings: float,
) -> SemanticCacheRepository:
    embedder = Embedder(provider or FakeProvider(), embedding_model)  # type: ignore[arg-type]
    return RedisSemanticCache(client, CacheSettings(**settings), embedder)  # type: ignore[arg-type]


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

    def test_undecodable_answer_misses(self, client: FakeRedis) -> None:
        cache = make_cache(client)
        cache.store("what is redis?", SCOPE, "An in-memory store.")
        next(iter(client.hashes.values()))["answer"] = b"\xff"

        assert cache.lookup("what is redis?", SCOPE) is None

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

    @pytest.mark.parametrize("field", ["embedding", "context_embedding"])
    def test_malformed_entry_is_a_miss(self, client: FakeRedis, field: str) -> None:
        cache = make_cache(client)
        cache.store("give me an example", SCOPE, "SET key value", context="what is redis?")
        for entry in client.hashes.values():
            entry[field] = b"abc"

        assert cache.lookup("give me an example", SCOPE, context="what is redis?") is None

    def test_embedding_failure_is_a_miss(self, client: FakeRedis) -> None:
        assert make_cache(client).lookup("embedding failure", SCOPE) is None

    def test_query_is_embedded_with_the_embedding_model(self, client: FakeRedis) -> None:
        provider = FakeProvider()

        make_cache(client, embedding_model="mxbai-embed-large", provider=provider).lookup("what is redis?", SCOPE)

        assert provider.calls == [(["what is redis?"], "mxbai-embed-large")]

    def test_query_and_context_are_embedded_in_a_single_call(self, client: FakeRedis) -> None:
        provider = FakeProvider()

        make_cache(client, provider=provider).lookup("give me an example", SCOPE, context="what is redis?")

        assert provider.calls == [(["give me an example", "what is redis?"], "nomic-embed-text")]


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

    def test_query_with_context_hits_an_entry_without_context(self, client: FakeRedis) -> None:
        cache = make_cache(client)
        cache.store("what is redis?", SCOPE, "An in-memory store.")

        assert cache.lookup("what is redis?", SCOPE, context="how to cook pasta?") == "An in-memory store."

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

    def test_text_that_is_not_valid_utf8_round_trips(self, client: FakeRedis) -> None:
        cache = make_cache(client)
        cache.store("what is redis?", SCOPE, "caf\udce9")

        assert cache.lookup("what is redis?", SCOPE) == "caf\udce9"
