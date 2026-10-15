import pytest

from sensai.commands.base import Command
from tests.commands.fakes import FakeCommand, FakeUI, MakeFakeCommand


@pytest.fixture
def fake_ui() -> FakeUI:
    ui = FakeUI()
    Command.bind_ui(ui)
    return ui


@pytest.fixture
def make_fake_command() -> MakeFakeCommand:
    def factory(name: str, help_text: str = "help text") -> FakeCommand:
        command_cls = type(f"FakeCommand_{name}", (FakeCommand,), {"name": name, "help": help_text})
        return command_cls()

    return factory
