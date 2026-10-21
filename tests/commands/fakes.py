from collections.abc import Callable
from typing import override
from unittest.mock import MagicMock

from sensai.commands.base import Command
from sensai.llm.message import Role
from sensai.memory.history import History
from sensai.ui.adapter import Confirmation, UIAdapter


class FakeUI(UIAdapter):
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.thinking_efforts: list[str] = []
        self.confirmations: list[Confirmation] = []
        self.confirmation_requests: list[tuple[str, dict]] = []
        self.answers: list[str | None] = []
        self.questions: list[str] = []

    @override
    def ask_confirmation(self, tool_name: str, arguments: dict) -> Confirmation:
        self.confirmation_requests.append((tool_name, arguments))
        return self.confirmations.pop(0) if self.confirmations else Confirmation.DENY

    @override
    def ask_user(self, question: str) -> str | None:
        self.questions.append(question)
        return self.answers.pop(0) if self.answers else None

    @override
    def open(self) -> bool:
        return True

    @override
    def run(self) -> None:
        pass

    @override
    def close(self) -> bool:
        return True

    @override
    def wait_event(self, *, timeout: float | None = None) -> bool:
        return False

    @override
    def get_event(self) -> None:
        return None

    @override
    def send_ai_response(self, response: object) -> bool:
        return True

    @override
    def send_error(self, message: str) -> bool:
        return True

    @override
    def send_input(self, role: Role, response: str, *, command: bool = False) -> bool:
        return True

    @override
    def load_conversation(self, messages: History) -> None:
        pass

    @override
    def send_system_message(self, message: str, *, append_response: bool = False) -> bool:
        self.messages.append(message)
        return True

    @override
    def set_completions(self, completions: dict[str, Callable[[list[str]], list[str]]]) -> bool:
        return True

    @override
    def update_thinking_effort(self, level: str) -> bool:
        self.thinking_efforts.append(level)
        return True


class FakeCommand(Command):
    def __init__(self) -> None:
        self.mock_execute = MagicMock()

    @override
    def execute(self, *args: str) -> None:
        self.mock_execute(*args)


type MakeFakeCommand = Callable[..., FakeCommand]
