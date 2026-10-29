from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import HorizontalGroup, VerticalGroup
from textual.markup import escape
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label

import sensai.ui.tui._effect as fx
from sensai.ui.adapter import CompressionChoice, Confirmation

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from sensai.monitoring.budget import ContextUsage

MAX_ARGUMENT_LEN = 200

DIALOG_CSS = """
    align: center middle;

    #dialog {
        width: 80%;
        max-width: 100;
        height: auto;
        padding: 1 2;
        border: round pink;
        background: $surface;
    }
    #dialog Label {
        width: 100%;
        margin-bottom: 1;
    }
    #buttons {
        width: 100%;
        height: auto;
        align: center middle;
    }
    #buttons Button {
        margin: 0 1;
    }
"""


def _format_arguments(arguments: dict) -> str:
    """List `arguments` one per line, cutting the long values so the dialog stays readable."""
    if not arguments:
        return fx.italic("(no arguments)")
    lines = []
    for key, value in arguments.items():
        text = str(value)
        if len(text) > MAX_ARGUMENT_LEN:
            text = text[:MAX_ARGUMENT_LEN] + "…"
        lines.append(f"{fx.bold(escape(str(key)))}: {escape(text)}")
    return "\n".join(lines)


class ConfirmScreen(ModalScreen[Confirmation]):
    """Asks whether a tool may run: once, for the rest of the session, or not at all."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("y", "answer('allow')", "Allow"),
        Binding("a", "answer('always')", "Always"),
        Binding("n", "answer('deny')", "Deny"),
        Binding("escape", "answer('deny')", "Deny", show=False),
    ]
    DEFAULT_CSS = f"ConfirmScreen {{ {DIALOG_CSS} }}"

    _ANSWERS: ClassVar[dict[str, Confirmation]] = {
        "allow": Confirmation.ALLOW,
        "always": Confirmation.ALWAYS,
        "deny": Confirmation.DENY,
    }

    def __init__(self, tool_name: str, arguments: dict) -> None:
        """Ask about the call of `tool_name` with `arguments`."""
        super().__init__()
        self._tool_name = tool_name
        self._arguments = arguments

    def compose(self) -> ComposeResult:
        """Build the question, the arguments of the call and the three answers."""
        with VerticalGroup(id="dialog"):
            yield Label(fx.bold(f"Allow the tool {escape(self._tool_name)} to run?"))
            yield Label(_format_arguments(self._arguments))
            with HorizontalGroup(id="buttons"):
                yield Button("Allow (y)", id="allow", variant="success")
                yield Button("Always this session (a)", id="always", variant="primary")
                yield Button("Deny (n)", id="deny", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Close the dialog with the answer of the button pressed."""
        self.action_answer(event.button.id or "deny")

    def action_answer(self, answer: str) -> None:
        """Close the dialog with `answer`, denying anything unexpected."""
        self.dismiss(self._ANSWERS.get(answer, Confirmation.DENY))


class CompressionScreen(ModalScreen[CompressionChoice]):
    """Asks how to go on with a conversation near its budget: summarise its oldest messages, or start anew."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("c", "answer('compress')", "Compress"),
        Binding("n", "answer('new')", "New conversation"),
        Binding("escape", "answer('compress')", "Compress", show=False),
    ]
    DEFAULT_CSS = f"CompressionScreen {{ {DIALOG_CSS} }}"

    _ANSWERS: ClassVar[dict[str, CompressionChoice]] = {
        "compress": CompressionChoice.COMPRESS,
        "new": CompressionChoice.NEW_CONVERSATION,
    }

    def __init__(self, usage: ContextUsage) -> None:
        """Ask about a conversation taking `usage` of its budget."""
        super().__init__()
        self._usage = usage

    def compose(self) -> ComposeResult:
        """Build the question and the two ways to go on."""
        share = "most" if self._usage.ratio is None else f"{self._usage.ratio:.0%}"
        with VerticalGroup(id="dialog"):
            yield Label(fx.bold(f"The conversation takes {share} of its context budget."))
            yield Label("Summarise its oldest messages to go on, or start a new conversation with this message?")
            with HorizontalGroup(id="buttons"):
                yield Button("Compress and continue (c)", id="compress", variant="primary")
                yield Button("New conversation (n)", id="new", variant="success")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Close the dialog with the answer of the button pressed."""
        self.action_answer(event.button.id or "compress")

    def action_answer(self, answer: str) -> None:
        """Close the dialog with `answer`, compressing on anything unexpected."""
        self.dismiss(self._ANSWERS.get(answer, CompressionChoice.COMPRESS))


class QuestionScreen(ModalScreen[str | None]):
    """Asks the user a question from the model, returning their answer, or None when they cancel."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel", show=False)]
    DEFAULT_CSS = f"QuestionScreen {{ {DIALOG_CSS} }}"

    def __init__(self, question: str) -> None:
        """Ask `question`."""
        super().__init__()
        self._question = question

    def compose(self) -> ComposeResult:
        """Build the question and the field the answer is typed in."""
        with VerticalGroup(id="dialog"):
            yield Label(fx.bold("Sensai asks:") + "\n" + escape(self._question))
            yield Input(id="answer", placeholder="Your answer (enter to send, escape to skip)")

    def on_mount(self) -> None:
        """Put the cursor in the answer field."""
        self.query_one("#answer", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Close the dialog with the answer, None when it is empty.

        The event is stopped so that the chat does not also read the answer as a new message.
        """
        event.stop()
        self.dismiss(event.value.strip() or None)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Keep the typed answer from opening the command completions of the chat."""
        event.stop()

    def action_cancel(self) -> None:
        """Close the dialog without an answer."""
        self.dismiss(None)
