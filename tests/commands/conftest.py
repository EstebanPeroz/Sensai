import pytest

from tests.commands.fakes import FakeCommand, FakeUI, MakeFakeCommand


@pytest.fixture
def fake_ui() -> FakeUI:
    return FakeUI()


@pytest.fixture
def make_fake_command(fake_ui: FakeUI) -> MakeFakeCommand:
    def factory(name: str, help_text: str = "help text") -> FakeCommand:
        command_cls = type(f"FakeCommand_{name}", (FakeCommand,), {"name": name, "help": help_text})
        return command_cls(fake_ui)

    return factory
