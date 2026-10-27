import threading
from typing import TYPE_CHECKING

from sensai.error import SensaiError
from sensai.llm.message import Message, Role
from sensai.memory.history import History
from sensai.memory.persistent_db import get_db
from sensai.pipeline import ChatContext
from sensai.semantic.cache_key import cache_key, cached_response

if TYPE_CHECKING:
    from collections.abc import Iterator
    from uuid import UUID

    from sensai.llm.persona_manager import PersonaManager
    from sensai.llm.provider_manager import ProviderManager
    from sensai.llm.responses import ChatResponse
    from sensai.pipeline import ChatPipeline
    from sensai.semantic.cache_key import CacheKey
    from sensai.semantic.cache_repository import SemanticCacheRepository
    from sensai.ui.adapter import UIAdapter


class Conversation:
    """Drives a chat exchange with a provider, streaming its response back to a UI."""

    _persona_manager: PersonaManager
    _provider_manager: ProviderManager
    _cache: SemanticCacheRepository | None
    history: History
    _pipeline: ChatPipeline
    uuid: UUID | None = None

    def __init__(
        self,
        provider_manager: ProviderManager,
        persona_manager: PersonaManager,
        pipeline: ChatPipeline,
        cache: SemanticCacheRepository | None = None,
    ) -> None:
        """Init the conversation on the `pipeline` it chats through, and the cache it is served from, if any."""
        self._pipeline = pipeline
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
        key: CacheKey,
    ) -> None:
        """Forward chunks to the UI until they run out or a stop is requested, then cache a complete answer.

        An answer built with tools is not cached: it relies on facts, such as web results, that may soon be outdated.
        """
        try:
            content: str = ""
            failed = False
            used_tools = False
            for chunk in chunks:
                if stop_streaming.is_set():
                    return
                if chunk.tool_calls:
                    used_tools = True
                if chunk.content is not None:
                    content += chunk.content
                ui.send_ai_response(chunk)
                if chunk.error is not None:
                    failed = True
                    break
            self.add_message(Role.ASSISTANT, content, ui)
            if (
                self._cache is not None
                and not failed
                and not used_tools
                # A blank answer, whitespace only, is not worth serving again.
                and content.strip() != ""
            ):
                self._cache.store(key.query, key.scope, content, key.context)
        except SensaiError as err:
            ui.send_error(str(err))
        finally:
            stream_done.set()

    def chat(self, user_input: str, ui: UIAdapter) -> None:
        """Send `user_input` to the provider and stream its response to `ui` until done or interrupted."""
        ui.send_input(Role.USER, user_input)
        persona = self._persona_manager.active_persona()
        context = ChatContext(message=user_input, ui=ui, messages=self.history.to_json())
        chunks = self._pipeline.execute(context)
        key = cache_key(user_input, self._provider_manager.provider().current_model(), persona, self.history)
        if self.add_message(Role.USER, user_input, ui) is False or self._reply_from_cache(key, ui):
            return

        stop_streaming = threading.Event()
        stream_done = threading.Event()
        threading.Thread(
            target=self._stream_chunks,
            args=(chunks, stop_streaming, stream_done, ui, key),
            daemon=True,
        ).start()

        while not stream_done.is_set():
            if not ui.wait_event(timeout=0.05):
                continue
            stop_streaming.set()
            break

    def _reply_from_cache(self, key: CacheKey, ui: UIAdapter) -> bool:
        """Send the cached answer to the query of `key` to the UI and save it, if any, and return whether there was."""
        response = cached_response(self._cache, key) if self._cache is not None else None
        if response is None:
            return False
        ui.send_ai_response(response)
        self.add_message(Role.ASSISTANT, response.content or "", ui)
        return True
