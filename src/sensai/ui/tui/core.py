from __future__ import annotations

import threading
from typing import TYPE_CHECKING, override

from sensai import ui
from sensai.ui.tui._dialogs import ConfirmScreen, QuestionScreen
from sensai.ui.tui._textual import ChatApp

if TYPE_CHECKING:
    from textual.screen import Screen

    from sensai.llm.message import Role
    from sensai.llm.responses import ChatResponse
    from sensai.memory.history import History
    from sensai.monitoring.metrics import Report
    from sensai.ui.completion import CompletionSource

DIALOG_POLL_SECONDS = 0.1
"""How often a thread waiting for a dialog checks that the app is still running."""


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

    def send_error(self, message: str) -> bool:
        """Send an error message to the UI."""
        try:
            self.app.call_from_thread(self.app.add_error, message)
        except RuntimeError:
            return False
        return True

    def send_input(self, role: Role, response: str, *, command: bool = False) -> bool:
        """Send a user conversation input to the UI."""
        try:
            self.app.call_from_thread(self.app.add_input, role, response, command=command)
        except RuntimeError:
            return False
        return True

    def load_conversation(self, messages: History) -> None:
        """Load conversation history to the ui."""
        try:
            self.app.call_from_thread(self.app.load_conversation, messages)
        except RuntimeError:
            return

    @override
    def send_system_message(self, message: str, *, append_response: bool = False) -> bool:
        """Send an application message to the UI."""
        try:
            self.app.call_from_thread(self.app.add_system_message, message, append_response=append_response)
        except RuntimeError:
            return False
        return True

    @override
    def set_completions(self, completions: dict[str, CompletionSource]) -> bool:
        """Give the commands and their completion functions to the input."""
        try:
            self.app.call_from_thread(self.app.set_completions, completions)
        except RuntimeError:
            return False
        return True

    @override
    def update_thinking_effort(self, level: str) -> bool:
        """Show the thinking effort `level` in the UI."""
        try:
            self.app.call_from_thread(self.app.update_thinking_effort, level)
        except RuntimeError:
            return False
        return True

    @override
    def show_dashboard(self, report: Report) -> bool:
        """Open the dashboard screen over the chat."""
        try:
            self.app.call_from_thread(self.app.show_dashboard, report)
        except RuntimeError:
            return False
        return True

    @override
    def ask_confirmation(self, tool_name: str, arguments: dict) -> ui.Confirmation:
        """Ask in a dialog whether `tool_name` may run, denying it when the dialog cannot be shown."""
        answer = self._wait_for_screen(ConfirmScreen(tool_name, arguments))
        return ui.Confirmation.DENY if answer is None else answer

    @override
    def ask_user(self, question: str) -> str | None:
        """Ask `question` in a dialog, then show the question and the answer in the chat."""
        answer = self._wait_for_screen(QuestionScreen(question))
        self.send_system_message(f"Sensai asked: {question}\n> {answer or '(no answer)'}")
        return answer

    def _wait_for_screen[T](self, screen: Screen[T]) -> T | None:
        """Show the modal `screen` and block the calling thread until it closes, returning its result.

        Returns None when the screen cannot be shown or the app stops before it closes, so the
        calling thread is never left waiting. The answer does not go through the event queue,
        where it would be read as a new message interrupting the current answer.
        """
        answered = threading.Event()
        result: list[T | None] = [None]

        def on_dismiss(value: T | None) -> None:
            result[0] = value
            answered.set()

        try:
            self.app.call_from_thread(self.app.open_dialog, screen, on_dismiss)
        except RuntimeError:
            return None
        while not answered.wait(DIALOG_POLL_SECONDS):
            if not self.app.is_running:
                return None
        return result[0]
