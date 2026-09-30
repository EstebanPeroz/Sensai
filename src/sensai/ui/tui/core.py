from __future__ import annotations

from typing import TYPE_CHECKING

from sensai import ui
from sensai.ui.tui._textual import ChatApp

if TYPE_CHECKING:
    from sensai.llm.responses import ChatResponse


class UITextualAdapter(ui.UIAdapter):
    """UIAdapter implementation backed by the Textual `ChatApp`.

    Bridges the Textual app, which owns the main thread and its own event
    loop, with the rest of the application running on a worker thread:
    calls into the app are marshalled via `call_from_thread`, and user
    input flows back through a shared `EventQueue`.
    """

    app: ChatApp
    _event_queue: ui.EventQueue

    def __init__(self) -> None:
        """Create the shared event queue and the underlying ChatApp instance."""
        super().__init__()
        self._event_queue = ui.EventQueue()
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

    def get_event(self) -> ui.Event | None:
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

    def send_system_message(self, message: str) -> bool:
        """Send an application message to the UI."""
        try:
            self.app.call_from_thread(self.app.add_system_message, message)
        except RuntimeError:
            return False
        return True
