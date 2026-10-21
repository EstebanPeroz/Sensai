from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import HorizontalGroup, VerticalGroup
from textual.markup import escape
from textual.screen import ModalScreen
from textual.widgets import Button, Label

import sensai.ui.tui._effect as fx
from sensai.ui.adapter import Confirmation

if TYPE_CHECKING:
    from textual.app import ComposeResult

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
