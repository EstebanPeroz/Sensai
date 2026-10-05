from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import HorizontalGroup
from textual.markup import escape
from textual.suggester import SuggestFromList
from textual.widgets import Input, Label, ListItem, ListView

import sensai.ui.tui._effect as fx
from sensai import ui
from sensai.llm.message import Role

if TYPE_CHECKING:
    from sensai.llm.responses import ChatResponse
    from sensai.memory.history import History

BINDINGS = [Binding("tab", "accept_suggestion", show=False, priority=True)]


USER_PROMPT_BACKGROUND: str = "grey"
"""Background color used for the "> " user prompt. Change this to restyle it.

Applied as CSS on the `ListItem` (see `ChatApp.CSS`), not as inline markup,
so it fills the full line width instead of just the text.
"""


def _format(mode: str, text: str) -> str:
    text = escape(text)
    match mode:
        case "message":
            formatted = "> " + text
        case "command":
            formatted = "> " + fx.italic(text)
        case "system":
            formatted = fx.italic(fx.bold("* system:")) + "\n" + text
        case "thinking":
            formatted = fx.grey("• " + text)
        case "content":
            formatted = fx.white("• " + text)
        case "error":
            formatted = fx.red("@ " + text)
        case _:
            formatted = text
    return formatted


class AppHeader(HorizontalGroup):
    """Top banner: the "Sensai" brand mark on the left, current directory on the right.

    The right side is its own container (`#header-right`) so further status
    widgets (model name, connection state, etc.) can be mounted into it
    later without reworking this layout.
    """

    def compose(self) -> ComposeResult:
        """Build the header's left brand and right status sections."""
        yield Label(fx.bold("Sensai"), id="header-brand")
        yield HorizontalGroup(Label(str(Path.cwd()), id="header-cwd"), id="header-right")


class ChatItem(ListItem):
    """A chat `ListView` entry tagged with its role and rendering mode.

    The role ("user"/"assistant"/"system"), rendering `mode`
    ("message"/"command"/"system"/"content"/"thinking"/"error") and accumulated raw text live
    as plain attributes on the widget itself, so the `ListView` is the only
    source of truth for conversation history. These attributes are never
    rendered; only the `Label`, re-rendered through `_format` on every
    change, is visible.
    """

    role: str
    mode: str
    text: str

    def __init__(self, role: str, mode: str, text: str) -> None:
        """Create a list item for the given role/mode, rendering `text`."""
        self.role = role
        self.mode = mode
        self.text = text
        classes = "user-message" if role == "user" else None
        super().__init__(HorizontalGroup(Label(_format(mode, text))), classes=classes)

    def update_text(self, text: str) -> None:
        """Append to the accumulated text and refresh the rendered label."""
        self.text += text
        self.query_one(Label).update(_format(self.mode, self.text))


class ChatApp(App):
    """Textual application implementing the chat UI.

    Renders the conversation as a scrolling list, streams AI responses into
    it, and forwards submitted user input to the shared `EventQueue` for the
    application's worker thread to consume.
    """

    CSS = f"""
    AppHeader {{
        height: 5%;
        width: 100%;
    }}
    AppHeader #header-brand {{
        width: 1fr;
        height: 100%;
        content-align: left middle;
    }}
    AppHeader #header-right {{
        width: 1fr;
        height: 100%;
        align: right middle;
    }}
    AppHeader #header-cwd {{
        width: auto;
        height: auto;
    }}
    ListView#chat {{
        width: 100%;
    }}
    ListView#chat > ListItem {{
        width: 100%;
    }}
    ListView#chat > ListItem HorizontalGroup {{
        width: 100%;
        height: auto;
    }}
    ListView#chat > ListItem Label {{
        width: 100%;
        height: auto;
    }}
    ListView#chat > ListItem.user-message Label {{
        background: {USER_PROMPT_BACKGROUND};
    }}
    ListView#chat > ListItem.user-message {{
        margin-top: 1;
    }}
    ListView#chat > ListItem.user-message:first-child {{
        margin-top: 0;
    }}
    Input#input {{
        height: 3;
        border: round pink;
        padding: 0 1;
    }}
    """

    _event_queue: ui.EventQueue
    ready: threading.Event

    def __init__(self, event_queue: ui.EventQueue) -> None:
        """Store the shared event queue and initialize conversation state."""
        super().__init__()
        self._event_queue = event_queue
        self.ready = threading.Event()

    def compose(self) -> ComposeResult:
        """Build the widget tree: a header, a scrolling chat list, and a text input for messages."""
        yield AppHeader()
        yield ListView(id="chat")
        yield Input(id="input", placeholder="Type a message...")

    def on_mount(self) -> None:
        """Signal that the app has finished mounting and is safe to talk to."""
        self.ready.set()

    def add_ai_response(self, resp: ChatResponse) -> None:
        """Append streamed text to the last chat entry, or start a new one."""
        if resp.error is not None:
            self._display_response("error", resp.error)
            return

        last = self._last_item()
        if last is None or last.mode == "thinking":
            if resp.thinking is not None:
                self._display_response("thinking", resp.thinking)
            if resp.content is not None:
                self._display_response("content", resp.content)
        else:
            if resp.content is not None:
                self._display_response("content", resp.content)
            if resp.thinking is not None:
                self._display_response("thinking", resp.thinking)

    def add_error(self, message: str) -> None:
        """Append an error message as a new entry in the chat."""
        self._display_response("error", message)

    def add_input(self, role: Role, user: str, *, command: bool = False) -> None:
        """Append a conversation message, or a typed `command`, as a new entry in the chat."""
        chat = self.query_one("#chat", ListView)
        if role == Role.USER:
            role_name, mode = "user", "command" if command else "message"
        elif role == Role.ASSISTANT:
            role_name, mode = "assistant", "content"
        else:
            return

        chat.append(ChatItem(role_name, mode, user))
        self._follow_scrolling()

    def add_system_message(self, message: str, *, append_response: bool) -> None:
        """Append an application message as a new entry in the chat."""
        chat = self.query_one("#chat", ListView)
        chat.append(ChatItem("system", "text" if append_response else "system", message))
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
            chat.append(ChatItem("assistant", mode, content))
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

    def load_conversation(self, messages: History) -> None:
        """Load conversation history to the ui."""
        chat = self.query_one("#chat", ListView)
        chat.clear()
        for message in messages.messages():
            self.add_input(message.role, message.content)

    def set_commands(self, commands: list[str]) -> None:
        """Suggest the given command names while the user types."""
        self.query_one("#input", Input).suggester = SuggestFromList(commands, case_sensitive=False)

    def action_accept_suggestion(self) -> None:
        """Accept the input's current suggestion."""
        self.query_one("#input", Input).action_cursor_right()
