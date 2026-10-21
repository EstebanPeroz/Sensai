import threading
from typing import TYPE_CHECKING, NamedTuple

from sensai.error import SensaiError
from sensai.llm.message import Message, Role
from sensai.llm.responses import ChatResponse
from sensai.memory.history import History
from sensai.memory.persistent_db import get_db
from sensai.memory.semantic_cache_repository import CacheScope

if TYPE_CHECKING:
    from collections.abc import Iterator
    from uuid import UUID

    from sensai.llm.persona_manager import PersonaManager
    from sensai.llm.provider_manager import ProviderManager
    from sensai.llm.thinking.react import ReAct
    from sensai.memory.semantic_cache_repository import SemanticCacheRepository
    from sensai.ui.adapter import UIAdapter

# Persona a cached answer is scoped to when none is active.
NO_PERSONA = ""
# Previous user messages a query is cached with: the same question can call for another answer in another context.
CACHE_CONTEXT_TURNS = 2


class CacheKey(NamedTuple):
    """What a query is looked up and cached with: the query itself, its scope and the previous user messages."""

    query: str
    scope: CacheScope
    context: str


class Conversation:
    """Drives a chat exchange with a provider, streaming its response back to a UI."""

    _react: ReAct
    _persona_manager: PersonaManager
    _provider_manager: ProviderManager
    _cache: SemanticCacheRepository | None
    history: History
    uuid: UUID | None = None

    def __init__(
        self,
        provider_manager: ProviderManager,
        persona_manager: PersonaManager,
        react: ReAct,
        cache: SemanticCacheRepository | None = None,
    ) -> None:
        """Init method to create the conversation callss, with the semantic cache answers are served from, if any."""
        self._react = react
        self._provider_manager = provider_manager
        self.history = History()
        self._persona_manager = persona_manager
        self._cache = cache

    def add_conv_to_db(self, uuid: UUID | None = None, name: str = "Branch") -> None:
        """Init the conversation in the DB if first message."""
        db = get_db()
        if db is None:
            return
        self.uuid = db.create_branch(uuid, name)
        if self.uuid:
            db.set_branch_model(self.uuid, self._provider_manager.provider().current_model())

    def add_message(self, role: Role, content: str, ui: UIAdapter) -> bool:
        """Add message to History and db."""
        db = get_db()
        if not self.uuid or not db:
            ui.send_system_message("Failed to add message")
            return False

        try:
            uuid = db.add_message_to_branch(self.uuid, role, content)
        except SensaiError as err:
            ui.send_system_message(err.message)
            return False

        if not uuid:
            ui.send_system_message("Failed to add message")
            return False

        self.history.insert(Message(uuid=uuid, role=role, content=content))
        return True

    def _stream_chunks(
        self,
        chunks: Iterator[ChatResponse],
        stop_streaming: threading.Event,
        stream_done: threading.Event,
        ui: UIAdapter,
        cache_key: CacheKey | None,
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
            self.add_message(Role.ASSISTANT, content, ui)
            if self._cache is not None and cache_key is not None and not failed and content.strip():
                self._cache.store(cache_key.query, cache_key.scope, content, cache_key.context)
        except SensaiError as err:
            ui.send_error(str(err))
        finally:
            stream_done.set()

    def chat(self, user_input: str, ui: UIAdapter) -> None:
        """Send `user_input` to the provider and stream its response to `ui` until done or interrupted."""
        ui.send_input(Role.USER, user_input)
        persona = self._persona_manager.active_persona()
        cache_key = self._cache_key(user_input, persona.name if persona is not None else NO_PERSONA)
        if self.add_message(Role.USER, user_input, ui) is False:
            return
        if cache_key is not None and self._answer_from_cache(cache_key, ui):
            return
        messages = self.history.to_json()
        if persona is not None:
            messages = [persona.to_payload(), *messages]
        chunks = self._react.generate_answer(messages)
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

    def _cache_key(self, query: str, persona: str) -> CacheKey | None:
        """Return what `query` is looked up and cached with, or None without a cache.

        The scope is read on every query, as /model and /persona change the model and persona mid-conversation.
        """
        if self._cache is None:
            return None
        previous = [message.content for message in self.history.messages() if message.role is Role.USER]
        scope = CacheScope(model=self._provider_manager.provider().current_model(), persona=persona)
        return CacheKey(query, scope, "\n".join(previous[-CACHE_CONTEXT_TURNS:]))

    def _answer_from_cache(self, key: CacheKey, ui: UIAdapter) -> bool:
        """Send the cached answer to the query of `key` to the UI, if there is one, and return whether there was."""
        if self._cache is None:
            return False
        answer = self._cache.lookup(key.query, key.scope, key.context)
        if answer is None:
            return False
        message = {"role": Role.ASSISTANT.value, "content": answer}
        ui.send_ai_response(ChatResponse({"model": key.scope.model, "done": True, "message": message}))
        self.add_message(Role.ASSISTANT, answer, ui)
        return True
