import threading
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

from sensai.error import SensaiError
from sensai.llm.message import Message, Role
from sensai.memory.history import History
from sensai.memory.persistent_db import get_db
from sensai.pipeline import ChatContext
from sensai.semantic.cache_key import cache_key, cached_response

if TYPE_CHECKING:
    from collections.abc import Generator
    from uuid import UUID

    from sensai.core.events import EventBus
    from sensai.llm.persona_manager import PersonaManager
    from sensai.llm.provider_manager import ProviderManager
    from sensai.llm.responses import ChatResponse
    from sensai.monitoring.interaction import Interaction
    from sensai.pipeline import ChatPipeline
    from sensai.semantic.cache_key import CacheKey
    from sensai.semantic.cache_repository import SemanticCacheRepository
    from sensai.ui.adapter import UIAdapter


class Conversation:
    """Drives a chat exchange with a provider, streaming its response back to a UI.

    Every call to the model, whatever its outcome, is emitted on `events` as an `Interaction`.
    """

    _persona_manager: PersonaManager
    _provider_manager: ProviderManager
    _events: EventBus
    _cache: SemanticCacheRepository | None
    history: History
    _pipeline: ChatPipeline
    uuid: UUID | None = None

    def __init__(
        self,
        provider_manager: ProviderManager,
        persona_manager: PersonaManager,
        pipeline: ChatPipeline,
        events: EventBus,
        cache: SemanticCacheRepository | None = None,
    ) -> None:
        """Init the conversation on the `pipeline` it chats through, the `events` it reports on, and its cache."""
        self._pipeline = pipeline
        self._provider_manager = provider_manager
        self.history = History()
        self._persona_manager = persona_manager
        self._events = events
        self._cache = cache

    def add_conv_to_db(self, uuid: UUID | None = None, name: str = "Branch") -> None:
        """Init the conversation in the DB if first message."""
        db = get_db()
        if db is None:
            return
        self.uuid = db.create_branch(uuid, name)
        if self.uuid:
            db.set_branch_model(self.uuid, self._provider_manager.provider().current_model())

    def reset(self) -> None:
        """Leave the current branch for a new conversation, created in the DB with its first message."""
        self.uuid = None
        self.history = History()

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

    def _emit_interaction(self, interaction: Interaction, *, persona: str | None, ui: UIAdapter) -> None:
        """Publish `interaction`, tagged with this branch and `persona`; a listener failing must not break the chat."""
        try:
            self._events.emit(replace(interaction, branch_id=self.uuid, persona=persona))
        except SensaiError as err:
            ui.send_system_message(f"Failed to log interaction: {err}")

    def _stream_chunks(
        self,
        chunks: Generator[ChatResponse],
        stop_streaming: threading.Event,
        stream_done: threading.Event,
        ui: UIAdapter,
        key: CacheKey,
    ) -> None:
        """Forward chunks to the UI until they run out or a stop is requested, then cache a complete answer.

        An answer built with tools is not cached: it relies on facts, such as web results, that may soon be outdated.
        `chunks` is closed on the way out, so that a stopped stream ends its pending calls to the model.
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
            try:
                chunks.close()
            finally:
                stream_done.set()

    def chat(self, user_input: str, ui: UIAdapter) -> None:
        """Send `user_input` to the provider and stream its response to `ui` until done or interrupted."""
        ui.send_input(Role.USER, user_input)
        persona = self._persona_manager.active_persona()
        key = cache_key(user_input, self._provider_manager.provider().current_model(), persona, self.history)
        if self.add_message(Role.USER, user_input, ui) is False or self._reply_from_cache(key, ui):
            return
        context = ChatContext(
            message=user_input,
            ui=ui,
            messages=self.history.to_json(),
            branch_id=self.uuid,
            on_interaction=partial(
                self._emit_interaction, persona=persona.name if persona is not None else None, ui=ui
            ),
        )
        chunks = self._pipeline.execute(context)

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
