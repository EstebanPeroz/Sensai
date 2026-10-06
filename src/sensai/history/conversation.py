import threading
from typing import TYPE_CHECKING

from sensai.error import SensaiError
from sensai.llm.message import Role
from sensai.llm.responses import ChatResponse
from sensai.memory.history import History
from sensai.memory.semantic_cache_repository import CacheScope

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sensai.llm.adapter import ProviderAdapter
    from sensai.memory.semantic_cache_repository import SemanticCacheRepository
    from sensai.ui.adapter import UIAdapter

DEFAULT_PERSONA = "default"
# Previous user messages a query is cached with: the same question can call for another answer in another context.
CACHE_CONTEXT_TURNS = 2


class Conversation:
    """Drives a chat exchange with a provider, streaming its response back to a UI."""

    _provider: ProviderAdapter
    _model: str
    _history: History
    _cache: SemanticCacheRepository | None
    _cache_scope: CacheScope

    def __init__(self, provider: ProviderAdapter, model: str, cache: SemanticCacheRepository | None = None) -> None:
        """Bind the conversation to the `provider` and `model` used to answer it, and to the cache, if any."""
        self._provider = provider
        self._model = model
        self._history = History()
        self._cache = cache
        self._cache_scope = CacheScope(model=model, persona=DEFAULT_PERSONA)

    def _stream_chunks(
        self,
        chunks: Iterator[ChatResponse],
        stop_streaming: threading.Event,
        stream_done: threading.Event,
        ui: UIAdapter,
        cache_key: tuple[str, str] | None,
    ) -> None:
        """Forward chunks to the UI until they run out or a stop is requested, then cache a complete answer."""
        try:
            content: str = ""
            failed = False
            for chunk in chunks:
                if stop_streaming.is_set():
                    return
                if chunk.content is not None:
                    content += chunk.content
                ui.send_ai_response(chunk)

                if chunk.error is not None:
                    failed = True
                    break
            self._history.append(Role.ASSISTANT, content)
            if self._cache is not None and cache_key is not None and not failed and content.strip():
                query, context = cache_key
                self._cache.store(query, self._cache_scope, content, context)
        except SensaiError as err:
            print(str(err))
            # log error implementation
        finally:
            stream_done.set()

    def chat(self, user_input: str, ui: UIAdapter) -> None:
        """Send `user_input` to the provider and stream its response to `ui` until done or interrupted."""
        ui.send_user_input(user_input)
        cache_key = (user_input, self._cache_context()) if self._cache is not None else None
        self._history.append(Role.USER, user_input)

        if cache_key is not None and self._answer_from_cache(*cache_key, ui):
            return

        chunks = self._provider.chat(
            {"model": self._model, "messages": self._history.to_json()},
            stream=True,
        )
        if chunks is None:
            return

        stop_streaming = threading.Event()
        stream_done = threading.Event()
        threading.Thread(
            target=self._stream_chunks,
            args=(chunks, stop_streaming, stream_done, ui, cache_key),
            daemon=True,
        ).start()

        while not stream_done.is_set():
            if not ui.wait_event(timeout=0.05):
                continue
            stop_streaming.set()
            break

    def _cache_context(self) -> str:
        """Return the last user messages of the history, which the next query is cached with."""
        previous = [message.content for message in self._history.messages() if message.role is Role.USER]
        return "\n".join(previous[-CACHE_CONTEXT_TURNS:])

    def _answer_from_cache(self, query: str, context: str, ui: UIAdapter) -> bool:
        """Send the cached answer to `query` to the UI, if there is one, and return whether there was."""
        if self._cache is None:
            return False
        answer = self._cache.lookup(query, self._cache_scope, context)
        if answer is None:
            return False
        message = {"role": Role.ASSISTANT.value, "content": answer}
        ui.send_ai_response(ChatResponse({"model": self._model, "done": True, "message": message}))
        self._history.append(Role.ASSISTANT, answer)
        return True
