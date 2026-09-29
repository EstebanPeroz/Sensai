from __future__ import annotations

from typing import TYPE_CHECKING

from sensai.ui.adapter import Event, UIAdapter
from sensai.ui.event_queue import EventQueue
from sensai.ui.tui._textual import ChatApp

if TYPE_CHECKING:
    from sensai.llm.responses import ChatResponse


class UITextualAdapter(UIAdapter):
    """Adapter to use to make a valid UI support."""

    app: ChatApp
    _event_queue: EventQueue

    def __init__(self) -> None:
        """Init the textual."""
        super().__init__()
        self._event_queue = EventQueue()
        self.app = ChatApp(self._event_queue)

    def open(self) -> bool:
        """Wait until the UI has started and is ready to receive calls."""
        self.app.ready.wait()
        return True

    def run(self) -> None:
        """Block running the UI. Must be called from the main thread.

        Textual installs signal handlers (SIGTSTP/SIGCONT) on startup, which
        Python only allows from the main thread of the main interpreter.
        """
        self.app.run()

    def close(self) -> bool:
        """Stop the UI.

        `call_from_thread` raises RuntimeError when called from the app's own
        thread, so exit directly in that case instead of going through it.
        """
        try:
            self.app.call_from_thread(self.app.exit)
        except RuntimeError:
            self.app.exit()
        return True

    def wait_event(self, *, timeout: float | None = None) -> bool:
        """Wait for an event. If timeout is None, wait indefinitely."""
        return self._event_queue.wait_event(timeout=timeout)

    def get_event(self) -> Event | None:
        """Get the first event in the event queue."""
        return self._event_queue.get_event()

    def send_ai_response(self, response: ChatResponse) -> bool:
        """Send a ai response to the UI."""
        try:
            self.app.call_from_thread(self.app.add_ai_response, response)
        except RuntimeError:
            return False
        return True

    def send_user_input(self, response: str) -> bool:
        """Send a user conversation input to the UI."""
        try:
            self.app.call_from_thread(self.app.add_user_input, response)
        except RuntimeError:
            return False
        return True
