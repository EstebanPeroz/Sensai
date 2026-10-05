import threading
from typing import TYPE_CHECKING

from sensai.error import SensaiError
from sensai.llm.message import Role
from sensai.memory.history import History

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sensai.llm.provider_manager import ProviderManager
    from sensai.llm.responses import ChatResponse
    from sensai.ui.adapter import UIAdapter


class Conversation:
    """Drives a chat exchange with a provider, streaming its response back to a UI."""

    _provider_manager: ProviderManager
    _history: History

    def __init__(self, provider_manager: ProviderManager) -> None:
        """Bind the conversation to the `provider` and `model` used to answer it."""
        self._provider_manager = provider_manager
        self._history = History()

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
            self._history.append(Role.ASSISTANT, content)
        except SensaiError as err:
            ui.send_error(str(err))
        finally:
            stream_done.set()

    def chat(self, user_input: str, ui: UIAdapter) -> None:
        """Send `user_input` to the provider and stream its response to `ui` until done or interrupted."""
        ui.send_input(Role.USER, user_input)
        self._history.append(Role.USER, user_input)

        chunks = self._provider_manager.provider().chat(
            {"model": self._provider_manager.provider().current_model(), "messages": self._history.to_json()},
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
