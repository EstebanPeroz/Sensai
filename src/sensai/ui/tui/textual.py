from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.containers import HorizontalGroup
from textual.widgets import Input, Label, ListItem, ListView

from sensai.ui.adapter import Event, EventType

if TYPE_CHECKING:
    import queue

    from sensai.llm.responses import ChatResponse


class ListConv:
    """TMP."""

    _list_conv: list[dict[str, str]]

    def __init__(self) -> None:
        """TMP."""
        self._list_conv = []

    def last_same(self, response: ChatResponse) -> bool:
        """TMP."""
        mode = self._list_conv[-1]["mode"]

        return self._list_conv[-1]["role"] == "assistant" and (
            (mode == "thinking" and response.thinking is not None)
            or (mode == "content" and response.content is not None)
        )

    def empty(self) -> bool:
        """TMP."""
        return len(self._list_conv) == 0

    def append(self, data: dict) -> None:
        """TMP."""
        self._list_conv.append(data)

    def append_content_to_last(self, content: str) -> None:
        """TMP."""
        if self.empty():
            return
        self._list_conv[-1]["content"] += content

    def last_content(self) -> str:
        """TMP."""
        if len(self._list_conv) != 0:
            return self._list_conv[-1]["content"]
        return ""


class ChatApp(App):
    """Init the textual."""

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

    _list: ListConv = ListConv()

    def __init__(self, event_queue: queue.Queue[Event]) -> None:
        """Init the textual."""
        super().__init__()
        self._event_queue = event_queue
        self._last_response_text = ""
        self.ready = threading.Event()

    def compose(self) -> ComposeResult:
        """Init the textual."""
        yield ListView(id="chat")
        yield Input(id="input", placeholder="Type a message...")

    def on_mount(self) -> None:
        """Signal that the app has finished mounting and is safe to talk to."""
        self.ready.set()

    def add_ai_response(self, text: ChatResponse) -> None:
        """Append streamed text to the last chat entry, or start a new one."""
        chat = self.query_one("#chat", ListView)

        if text.content is not None:
            content = text.content
            mode = "content"
        elif text.thinking is not None:
            content = "[red]" + text.thinking + "[/red]"
            mode = "thinking"
        else:
            return

        if self._list.empty() or self._list.last_same(text) is False:
            data = {"role": "assistant", "mode": mode, "content": content}
            self._list.append(data)
            chat.append(ListItem(HorizontalGroup(Label(self._list.last_content()))))
        else:
            self._list.append_content_to_last(content)
            last_item = chat.children[-1]
            label = last_item.query_one(Label)
            label.update(self._list.last_content())

    def add_user_input(self, user: str) -> None:
        """TMP."""
        chat = self.query_one("#chat", ListView)
        self._list.append({"role": "user", "mode": "message", "content": user})
        chat.append(ListItem(HorizontalGroup(Label("[bold]* user: [/bold]" + user))))

    def on_input_submitted(self, message: Input.Submitted) -> None:
        """Handle a user submitting the input widget."""
        content = message.value.strip()
        message.input.clear()
        if not content:
            return
        event = Event()
        event.type = EventType.UserContent
        event.content = content
        self._event_queue.put_nowait(event)
