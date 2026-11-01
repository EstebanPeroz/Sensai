from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from sensai_server.cache.semantic_cache import SemanticCache


class SessionCache:
    """The view one session has of the semantic cache: its own entries, and no other session's.

    It remembers the vector of the last prompt it looked up, so that storing the answer to that same prompt does not
    embed it again.
    """

    _cache: SemanticCache
    _client_id: UUID
    _last: tuple[str, list[float]] | None

    def __init__(self, cache: SemanticCache, client_id: UUID) -> None:
        """Bind the view to the session `client_id` of `cache`."""
        self._cache = cache
        self._client_id = client_id
        self._last = None

    async def lookup(self, prompt: str) -> str | None:
        """Return the answer to this session's cached prompt closest to `prompt`, if it is close enough, else None.

        Raises:
            EmbeddingError: If the prompt could not be embedded.
            CacheError: If Redis fails.

        """
        vector = await self._cache.embed(prompt)
        self._last = (prompt, vector)
        closest = await self._cache.closest(self._client_id, vector)
        if closest is None:
            return None
        answer, similarity = closest
        return answer if similarity >= self._cache.similarity_threshold else None

    async def store(self, prompt: str, answer: str) -> None:
        """Cache `answer` to `prompt` for this session, reusing the vector `lookup` computed for the same prompt.

        Raises:
            EmbeddingError: If the prompt could not be embedded.
            CacheError: If Redis fails.

        """
        if self._last is not None and self._last[0] == prompt:
            vector = self._last[1]
        else:
            vector = await self._cache.embed(prompt)
        self._last = None
        await self._cache.store_entry(self._client_id, prompt, answer, vector)

    async def clear(self) -> None:
        """Delete every entry of this session.

        Call it whenever the session leaves its conversation: on `/branch set` and `/branch fork`, on close, and in
        any future command replacing `session.conv`. An answer cached in one conversation must never be reused in
        another.

        Raises:
            CacheError: If Redis fails.

        """
        self._last = None
        await self._cache.clear_session(self._client_id)
