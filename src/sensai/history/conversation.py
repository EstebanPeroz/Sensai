import threading
from typing import TYPE_CHECKING

from sensai.error import SensaiError
from sensai.llm.message import Message, Role
from sensai.memory.history import History
from sensai.memory.persistent_db import get_db

if TYPE_CHECKING:
    from collections.abc import Iterator
    from uuid import UUID

    from sensai.llm.provider_manager import ProviderManager
    from sensai.llm.responses import ChatResponse
    from sensai.ui.adapter import UIAdapter


class Conversation:
    """Drives a chat exchange with a provider, streaming its response back to a UI."""

    _provider_manager: ProviderManager
    history: History
    uuid: UUID | None = None

    def __init__(self, provider_manager: ProviderManager) -> None:
        """Bind the conversation to the `provider` and `model` used to answer it."""
        self._provider_manager = provider_manager
        self.history = History()

    def add_conv_to_db(self, uuid: UUID | None = None, name: str = "Branch") -> None:
        """Tmp."""
        db = get_db()
        if db is None:
            return
        self.uuid = db.create_branch(uuid, name)
        if self.uuid:
            db.set_branch_model(self.uuid, self._provider_manager.provider().current_model())

    def add_message(self, role: Role, content: str, ui: UIAdapter) -> bool:
        """Tp."""
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
    ) -> None:
        """Forward chunks to the UI until they run out or a stop is requested."""
        try:
            content: str = ""
            for chunk in chunks:
                if stop_streaming.is_set():
                    return
                if chunk.content is not None:
                    content += chunk.content
                ui.send_ai_response(chunk)

                if chunk.error is not None:
                    break
            self.add_message(Role.ASSISTANT, content, ui)
        except SensaiError as err:
            ui.send_error(str(err))
        finally:
            stream_done.set()

    def chat(self, user_input: str, ui: UIAdapter) -> None:
        """Send `user_input` to the provider and stream its response to `ui` until done or interrupted."""
        ui.send_input(Role.USER, user_input)
        if self.add_message(Role.USER, user_input, ui) is False:
            return

        chunks = self._provider_manager.provider().chat(
            {"model": self._provider_manager.provider().current_model(), "messages": self.history.to_json()},
            stream=True,
        )
        if chunks is None:
            return

        stop_streaming = threading.Event()
        stream_done = threading.Event()
        threading.Thread(
            target=self._stream_chunks,
            args=(chunks, stop_streaming, stream_done, ui),
            daemon=True,
        ).start()

        while not stream_done.is_set():
            if not ui.wait_event(timeout=0.05):
                continue
            stop_streaming.set()
            break
