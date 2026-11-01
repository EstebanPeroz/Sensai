from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

from sensai_protocol import (
    AiChunk,
    AskCompression,
    AskUser,
    Cancel,
    Chat,
    Command,
    Complete,
    Completions,
    CompressionAnswer,
    CompressionChoice,
    ConfirmAnswer,
    Confirmation,
    ConfirmTool,
    ContextUsage,
    Dashboard,
    Effort,
    Error,
    History,
    Login,
    Logout,
    RequestCommand,
    Role,
    Session,
    System,
    UserAnswer,
)
from sensai_protocol import Input as InputMessage
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import HorizontalGroup
from textual.markup import escape
from textual.widgets import Input, Label, ListItem, ListView, OptionList

import sensai_tui.ui._style as fx
from sensai_tui.client_commands.dispatch import run_client_command
from sensai_tui.completion import Completer
from sensai_tui.ui._dashboard import DashboardScreen
from sensai_tui.ui._dialogs import CompressionScreen, ConfirmScreen, QuestionScreen

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from sensai_protocol import ClientMessage, HistoryMessage, ServerMessage
    from textual.screen import Screen

    from sensai_tui.connection import Connection


def _tokens(count: int) -> str:
    """Write a token `count` short, in thousands from a thousand on (`12.3k`)."""
    return str(count) if count < 1000 else f"{count / 1000:.1f}k"  # noqa: PLR2004


def _format_usage(usage: ContextUsage) -> str:
    """Render the context `usage` as `context: used / budget (share)`, in red once near the limit, `?` when unknown."""
    used = "?" if usage.used is None else _tokens(usage.used)
    if usage.budget is None or usage.used is None or usage.ratio is None:
        return f"context: {used}" if usage.budget is None else f"context: {used} / {_tokens(usage.budget)}"
    text = f"context: {_tokens(usage.used)} / {_tokens(usage.budget)} ({usage.ratio:.0%})"
    return fx.red(text) if usage.near_limit else text


class CompletionList(OptionList, can_focus=False):
    """Dropdown of the completions of the input, driven from the input so it never takes the focus."""


class AppHeader(HorizontalGroup):
    """Top banner: the "Sensai" brand mark on the left, the model and current directory on the right.

    The right side is its own container (`#header-right`) so further status
    widgets (connection state, etc.) can be mounted into it later without
    reworking this layout.
    """

    def compose(self) -> ComposeResult:
        """Build the header's left brand and right status sections."""
        yield Label(fx.bold("Sensai"), id="header-brand")
        yield HorizontalGroup(Label("", id="header-model"), Label(str(Path.cwd()), id="header-cwd"), id="header-right")


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
    _label: Label

    def __init__(self, role: str, mode: str, text: str) -> None:
        """Create a list item for the given role/mode, rendering `text`."""
        self.role = role
        self.mode = mode
        self.text = text
        classes = "user-message" if role == "user" else None
        self._label = Label(fx.format_message(mode, text))
        super().__init__(HorizontalGroup(self._label), classes=classes)

    def update_text(self, text: str) -> None:
        """Append to the accumulated text and refresh the rendered label."""
        self.text += text
        self._label.update(fx.format_message(self.mode, self.text))


class ChatApp(App):
    """Textual application implementing the chat client.

    Sends what the user submits to the server as protocol messages, and
    renders the conversation from the messages the server sends back: the
    transcript only shows what the server received and answered.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("tab", "accept_completion", show=False, priority=True),
        Binding("up", "move_completion(-1)", show=False, priority=True),
        Binding("down", "move_completion(1)", show=False, priority=True),
        Binding("escape", "escape", show=False, priority=True),
        Binding("ctrl+c", "cancel", show=False, priority=True),
    ]
    AUTO_FOCUS = "#input"

    CSS = fx.TEXTURAL_CSS

    client_id: UUID | None
    _connection: Connection
    _model: str
    _completer: Completer
    _outbox: asyncio.Queue[ClientMessage]
    _connected: bool
    _streaming: bool

    def __init__(self, connection: Connection, model: str) -> None:
        """Store the connection to the server and the `model` to log in with."""
        super().__init__()
        self.client_id = None
        self._connection = connection
        self._model = model
        self._completer = Completer(lambda text: self._send(Complete(text=text)))
        self._outbox = asyncio.Queue()
        self._connected = True
        self._streaming = False

    def compose(self) -> ComposeResult:
        """Build the widget tree: a header, a scrolling chat list, and a text input for messages."""
        yield AppHeader()
        yield ListView(id="chat")
        yield CompletionList()
        yield Input(id="input", placeholder="Type a message...")

    def on_mount(self) -> None:
        """Start the workers that send the outgoing messages and receive the server's."""
        self.run_worker(self._sender(), name="sender")
        self.run_worker(self._receiver(), name="receiver")

    async def on_unmount(self) -> None:
        """Close the connection when the app stops."""
        await self._connection.close()

    async def _sender(self) -> None:
        """Send the queued messages to the server, one at a time and in order."""
        while True:
            message = await self._outbox.get()
            try:
                await self._connection.send(message)
            finally:
                self._outbox.task_done()

    async def _receiver(self) -> None:
        """Log in, then render each server message until the connection closes."""
        self._send(Login(model=self._model))
        async for message in self._connection.messages():
            self._on_server_message(message)
        self._set_disconnected()

    def _send(self, message: ClientMessage) -> None:
        """Queue `message` for the server. Dropped once disconnected."""
        if self._connected:
            self._outbox.put_nowait(message)

    def _set_disconnected(self) -> None:
        """Show that the server is gone and stop accepting input."""
        self._connected = False
        self._streaming = False
        self.add_error("disconnected from server")
        field = self.query_one("#input", Input)
        field.disabled = True
        field.placeholder = "disconnected — ctrl+q to quit"
        self.query_one(CompletionList).display = False
        while len(self.screen_stack) > 1:
            self.pop_screen()

    async def _quit(self) -> None:
        """Wait for the queued messages to be sent, then close the connection and exit."""
        await self._outbox.join()
        await self._connection.close()
        self.exit()

    def _on_server_message(self, message: ServerMessage) -> None:
        """Render a message received from the server."""
        match message:
            case Session():
                self.client_id = message.client_id
                self._completer.set_commands(message.commands)
                self.update_thinking_effort(message.effort)
                self.query_one("#header-model", Label).update(escape(message.model))
            case AiChunk():
                self.add_ai_response(message)
                self._streaming = not message.done and message.error is None
            case InputMessage():
                self.add_input(message.role, message.content, command=message.command)
            case System():
                self.add_system_message(message.message, append_response=message.append)
            case Error():
                self.add_error(message.message)
            case History():
                self.load_conversation(message.messages)
            case Effort() | ContextUsage():
                self._update_status(message)
            case Completions():
                self._receive_completions(message)
            case ConfirmTool() | AskUser() | AskCompression() | Dashboard() | RequestCommand():
                self._on_request(message)

    def _update_status(self, message: Effort | ContextUsage) -> None:
        """Show the thinking effort or the context usage on the border of the input box."""
        match message:
            case Effort():
                self.update_thinking_effort(message.level)
            case ContextUsage():
                self.update_context_usage(message)

    async def _run_client_command(self, message: RequestCommand) -> None:
        """Run the client command the server asked for, and send its answer back."""
        self._send(await run_client_command(message))

    def _receive_completions(self, message: Completions) -> None:
        """Show the received completions, unless they answer an outdated input."""
        if self._completer.receive(message.text, message.values):
            self._refresh_completions(self.query_one("#input", Input).value)

    def _on_request(self, message: ConfirmTool | AskUser | AskCompression | Dashboard | RequestCommand) -> None:
        """Open the dialog or the dashboard the server asked for, or run the client command it asked for."""
        match message:
            case RequestCommand():
                self.run_worker(self._run_client_command(message))
            case ConfirmTool():
                self.ask_confirmation(message)
            case AskUser():
                self.ask_user(message)
            case AskCompression():
                self.ask_compression(message)
            case Dashboard():
                self.push_screen(DashboardScreen(message.report))

    def ask_confirmation(self, message: ConfirmTool) -> None:
        """Ask in a dialog whether the tool may run, sending the answer back, deny when the dialog is closed."""

        def on_dismiss(answer: Confirmation | None) -> None:
            self._send(ConfirmAnswer(request_id=message.request_id, answer=answer or Confirmation.DENY))

        self.open_dialog(ConfirmScreen(message.tool_name, message.arguments), on_dismiss)

    def ask_user(self, message: AskUser) -> None:
        """Ask the model's question in a dialog, sending the answer back, None when it is skipped."""

        def on_dismiss(answer: str | None) -> None:
            self._send(UserAnswer(request_id=message.request_id, answer=answer))

        self.open_dialog(QuestionScreen(message.question), on_dismiss)

    def ask_compression(self, message: AskCompression) -> None:
        """Ask in a dialog how to go on with a conversation near its budget, compress when the dialog is closed."""

        def on_dismiss(answer: CompressionChoice | None) -> None:
            self._send(CompressionAnswer(request_id=message.request_id, answer=answer or CompressionChoice.COMPRESS))

        self.open_dialog(CompressionScreen(message), on_dismiss)

    def open_dialog[T](self, screen: Screen[T], on_dismiss: Callable[[T | None], None]) -> None:
        """Show the modal `screen` over the chat, `on_dismiss` receiving its result once it closes.

        The completion list is hidden first, so that escape reaches the dialog.
        """
        self.query_one(CompletionList).display = False
        self.push_screen(screen, on_dismiss)

    def add_ai_response(self, resp: AiChunk) -> None:
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
        """Send the submitted input to the server: `/q` logs out, `/...` is a command, anything else a chat.

        The inputs of dialogs are ignored.
        """
        if message.input.id != "input":
            return
        content = message.value.strip()
        message.input.clear()
        if not content or not self._connected:
            return
        if content == "/q":
            self._send(Logout())
            self.run_worker(self._quit(), name="quit")
            return
        if self._streaming:
            self._send(Cancel())
            self._streaming = False
        if content.startswith("/"):
            self._send(Command(command=content))
        else:
            self._send(Chat(prompt=content))

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

    def load_conversation(self, messages: list[HistoryMessage]) -> None:
        """Replace the transcript with a loaded conversation."""
        chat = self.query_one("#chat", ListView)
        chat.clear()
        for message in messages:
            self.add_input(message.role, message.content)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Show the completions of the new input value, or hide the list when there are none."""
        if event.input.id != "input":
            return
        self._refresh_completions(event.value)

    def _refresh_completions(self, value: str) -> None:
        """Show the completions of `value`, or hide the list when there are none."""
        dropdown = self.query_one(CompletionList)
        candidates = self._completer.complete(value)
        if candidates in ([], [value]):
            dropdown.display = False
            return
        dropdown.set_options(candidates)
        dropdown.highlighted = 0
        dropdown.display = True

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Accept a completion clicked with the mouse."""
        self._accept(str(event.option.prompt))

    @override
    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Only let the completion keys act while the list is shown, so they keep their usual role otherwise.

        Escape and cancel only act on the chat: over a dialog or the dashboard, they reach that screen.
        """
        if action in {"accept_completion", "move_completion"}:
            return self.query_one(CompletionList).display
        if action in {"escape", "cancel"}:
            return len(self.screen_stack) == 1
        return True

    def action_move_completion(self, step: int) -> None:
        """Move the highlight `step` options down (negative to go up)."""
        dropdown = self.query_one(CompletionList)
        if step < 0:
            dropdown.action_cursor_up()
        else:
            dropdown.action_cursor_down()

    def action_escape(self) -> None:
        """Close the completion list if shown, otherwise stop the running answer."""
        dropdown = self.query_one(CompletionList)
        if dropdown.display:
            dropdown.display = False
        else:
            self.action_cancel()

    def action_cancel(self) -> None:
        """Ask the server to stop the task running on the session."""
        self._send(Cancel())
        self._streaming = False

    def action_accept_completion(self) -> None:
        """Put the highlighted completion in the input."""
        dropdown = self.query_one(CompletionList)
        option = dropdown.highlighted_option
        if dropdown.display and option is not None:
            self._accept(str(option.prompt))

    def _accept(self, value: str) -> None:
        """Replace the input with `value`, cursor at the end, and give it back the focus."""
        field = self.query_one("#input", Input)
        field.value = value
        field.cursor_position = len(value)
        field.focus()

    def update_context_usage(self, usage: ContextUsage) -> None:
        """Show the context `usage` at the top left of the input box."""
        self.query_one("#input", Input).border_title = _format_usage(usage)

    def update_thinking_effort(self, level: str) -> None:
        """Show the thinking effort `level` at the bottom right of the input box."""
        self.query_one("#input", Input).border_subtitle = f"effort: {level}"
