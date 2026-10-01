from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.containers import HorizontalGroup
from textual.widgets import Input, Label, ListItem, ListView

import sensai.ui.tui._effect as fx
from sensai import ui

if TYPE_CHECKING:
    from sensai.llm.responses import ChatResponse


class ChatItem(ListItem):
    """A chat `ListView` entry tagged with its role and rendering mode.

    The role ("user"/"assistant"), rendering `mode`
    ("message"/"content"/"thinking"/"error") and accumulated text live as
    plain attributes on the widget itself, so the `ListView` is the only
    source of truth for conversation history. These attributes are never
    rendered; only the `Label` text is visible.
    """

    role: str
    mode: str
    text: str

    def __init__(self, role: str, mode: str, text: str) -> None:
        """Create a list item for the given role/mode, rendering `text`."""
        self.role = role
        self.mode = mode
        self.text = text
        super().__init__(HorizontalGroup(Label(text)))

    def update_text(self, text: str) -> None:
        """Append to the accumulated text and refresh the rendered label."""
        self.text += text
        self.query_one(Label).update(self.text)


class ChatApp(App):
    """Textual application implementing the chat UI.

    Renders the conversation as a scrolling list, streams AI responses into
    it, and forwards submitted user input to the shared `EventQueue` for the
    application's worker thread to consume.
    """

    CSS = """
    ListView#chat {
        width: 100%;
    }
    ListView#chat > ListItem {
        width: 100%;
    }
    ListView#chat > ListItem HorizontalGroup {
        height: auto;
    }
    ListView#chat > ListItem Label {
        width: 100%;
        height: auto;
    }
    """

    _event_queue: ui.EventQueue
    ready: threading.Event

    def __init__(self, event_queue: ui.EventQueue) -> None:
        """Store the shared event queue and initialize conversation state."""
        super().__init__()
        self._event_queue = event_queue
        self.ready = threading.Event()

    def compose(self) -> ComposeResult:
        """Build the widget tree: a scrolling chat list and a text input for messages."""
        yield ListView(id="chat")
        yield Input(id="input", placeholder="Type a message...")

    def on_mount(self) -> None:
        """Signal that the app has finished mounting and is safe to talk to."""
        self.ready.set()

    def add_ai_response(self, resp: ChatResponse) -> None:
        """Append streamed text to the last chat entry, or start a new one."""
        if resp.error is not None:
            self._display_response("error", fx.red(resp.error))
            return

        last = self._last_item()
        if last is None or last.mode == "thinking":
            if resp.thinking is not None:
                self._display_response("thinking", fx.grey(resp.thinking))
            if resp.content is not None:
                self._display_response("content", resp.content)
        else:
            if resp.content is not None:
                self._display_response("content", resp.content)
            if resp.thinking is not None:
                self._display_response("thinking", fx.grey(resp.thinking))

    def add_user_input(self, user: str) -> None:
        """Append the user's submitted message as a new entry in the chat."""
        chat = self.query_one("#chat", ListView)
        chat.append(ChatItem("user", "message", fx.bold("* user: ") + user))
        self._follow_scrolling()

    def on_input_submitted(self, message: Input.Submitted) -> None:
        """Handle a user submitting the input widget."""
        content = message.value.strip()
        message.input.clear()
        if not content:
            return
        self._event_queue.put(content)

    def _display_response(self, mode: str, content: str) -> None:
        """Render a chunk of an AI response in the chat.

        Appends to the last list item if it continues the same assistant
        `mode` (e.g. streamed "content" or "thinking" text), otherwise
        starts a new labeled list item.
        """
        chat = self.query_one("#chat", ListView)
        last = self._last_item()

        if last is None or not (last.role == "assistant" and last.mode == mode):
            chat.append(ChatItem("assistant", mode, fx.bold("* " + mode + ": ") + content))
        else:
            last.update_text(content)

        self._follow_scrolling()

    def _last_item(self) -> ChatItem | None:
        """Return the most recently appended chat item, or None if the chat is empty."""
        if len(self.query_one("#chat", ListView).children) == 0:
            return None
        return self.query_one("#chat", ListView).children[-1]  # type: ignore[return-value]

    def _follow_scrolling(self) -> None:
        """Auto-scroll the chat list to the bottom, but only if the user was already at the end."""
        chat = self.query_one("#chat", ListView)

        if chat.is_vertical_scroll_end:
            chat.scroll_end(animate=False)
