from __future__ import annotations

from typing import TYPE_CHECKING

import redis

from sensai.semantic.cache import RedisSemanticCache

if TYPE_CHECKING:
    from sensai.config.settings import CacheSettings
    from sensai.llm.embedder import Embedder
    from sensai.semantic.cache_repository import SemanticCacheRepository

REDIS_TIMEOUT = 1


def connect_cache(settings: CacheSettings, embedder: Embedder) -> SemanticCacheRepository | None:
    """Build the semantic cache, or None if its Redis server is unreachable: answers are then not cached."""
    client = redis.Redis.from_url(
        settings.redis_url, socket_connect_timeout=REDIS_TIMEOUT, socket_timeout=REDIS_TIMEOUT
    )
    try:
        client.ping()
    except redis.RedisError:
        client.close()
        return None
    return RedisSemanticCache(client, settings, embedder)
