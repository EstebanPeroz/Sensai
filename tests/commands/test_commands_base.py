from typing import override

import pytest

from sensai.commands.base import Command
from tests.commands.fakes import FakeUI, MakeFakeCommand


class TestCommand:
    def test_cannot_instantiate_directly(self, fake_ui: FakeUI) -> None:
        with pytest.raises(TypeError):
            Command()  # type: ignore[abstract]

    def test_subclass_without_execute_cannot_be_instantiated(self, fake_ui: FakeUI) -> None:
        class Incomplete(Command):
            name = "incomplete"
            help = "Missing execute."

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_implementing_execute_can_be_instantiated(self, fake_ui: FakeUI) -> None:
        class Concrete(Command):
            name = "concrete"
            help = "A concrete command."

            @override
            def execute(self, *args: str) -> None:
                self.called_with = args

        command = Concrete()
        command.execute("a", "b")

        assert command.called_with == ("a", "b")

    def test_stores_ui(self, fake_ui: FakeUI) -> None:
        class Concrete(Command):
            name = "concrete"
            help = "A concrete command."

            @override
            def execute(self, *args: str) -> None:
                self._ui.send_system_message("done")

        Concrete().execute()

        assert fake_ui.messages == ["done"]

    def test_has_no_completions_by_default(self, make_fake_command: MakeFakeCommand) -> None:
        assert make_fake_command("foo").completions([]) == []
