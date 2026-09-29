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


class ListConv:
    """In-memory history of chat entries backing the chat `ListView`.

    Each entry records its role ("user"/"assistant"), rendering `mode`
    ("message"/"content"/"thinking"/"error") and accumulated text, so the
    UI can decide whether new streamed content should be appended to the
    last list item or start a new one.
    """

    _list_conv: list[dict[str, str]]

    def __init__(self) -> None:
        """Initialize an empty conversation history."""
        self._list_conv = []

    def last_same(self, mode: str) -> bool:
        """Check whether the last entry is an assistant entry in the given rendering `mode`.

        Used to decide whether new content should be merged into the last
        list item instead of starting a new one.
        """
        last_mode = self._list_conv[-1]["mode"]

        return self._list_conv[-1]["role"] == "assistant" and (last_mode == mode)

    def empty(self) -> bool:
        """Return True if no entries have been recorded yet."""
        return len(self._list_conv) == 0

    def append(self, data: dict) -> None:
        """Record a new entry at the end of the conversation history."""
        self._list_conv.append(data)

    def last(self) -> dict | None:
        """Return the most recently recorded entry, or None if the history is empty."""
        if len(self._list_conv) != 0:
            return self._list_conv[-1]
        return None


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

    _list: ListConv
    _event_queue: ui.EventQueue
    ready: threading.Event

    def __init__(self, event_queue: ui.EventQueue) -> None:
        """Store the shared event queue and initialize conversation state."""
        super().__init__()
        self._event_queue = event_queue
        self.ready = threading.Event()
        self._list = ListConv()

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

        last = self._list.last()
        if last is None or last["mode"] == "thinking":
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
        self._list.append({"role": "user", "mode": "message", "content": user})
        chat.append(ListItem(HorizontalGroup(Label(fx.bold("* user: ") + user))))
        self._follow_scrolling()

    def on_input_submitted(self, message: Input.Submitted) -> None:
        """Handle a user submitting the input widget."""
        content = message.value.strip()
        message.input.clear()
        if not content:
            return
        event = ui.Event()
        event.type = ui.EventType.UserContent
        event.content = content
        self._event_queue.put(event)

    def _display_response(self, mode: str, content: str) -> None:
        """Render a chunk of an AI response in the chat.

        Appends to the last list item if it continues the same assistant
        `mode` (e.g. streamed "content" or "thinking" text), otherwise
        starts a new labeled list item.
        """
        chat = self.query_one("#chat", ListView)

        if self._list.empty() or self._list.last_same(mode) is False:
            content = fx.bold("* " + mode + ": ") + content
            data = {"role": "assistant", "mode": mode, "content": content}
            self._list.append(data)
            chat.append(ListItem(HorizontalGroup(Label(content))))
        else:
            last = self._list.last()
            if last is None:
                return
            last["content"] += content
            last_item = chat.children[-1]
            label = last_item.query_one(Label)
            label.update(last["content"])

        self._follow_scrolling()

    def _follow_scrolling(self) -> None:
        """Auto-scroll the chat list to the bottom, but only if the user was already at the end."""
        chat = self.query_one("#chat", ListView)

        if chat.is_vertical_scroll_end:
            chat.scroll_end(animate=False)
