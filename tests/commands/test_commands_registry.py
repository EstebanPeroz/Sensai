import pytest

from sensai.commands.registry import CommandRegistry, UnknownCommandError
from tests.commands.fakes import MakeFakeCommand


class TestUnknownCommandError:
    def test_stores_command_name_and_message(self) -> None:
        error = UnknownCommandError("nope")

        assert error.command_name == "nope"
        assert str(error) == "Unknown command: nope"


class TestCommandRegistryInit:
    def test_indexes_commands_by_name(self, make_fake_command: MakeFakeCommand) -> None:
        foo = make_fake_command("foo")
        bar = make_fake_command("bar")

        registry = CommandRegistry(commands=[foo, bar])

        assert sorted(registry.commands(), key=lambda c: c.name) == [bar, foo]

    def test_empty_registry_has_no_commands(self) -> None:
        registry = CommandRegistry(commands=[])

        assert registry.commands() == []


class TestRegisterCommand:
    def test_adds_new_command(self, make_fake_command: MakeFakeCommand) -> None:
        registry = CommandRegistry(commands=[])
        foo = make_fake_command("foo")

        registry.register_command(foo)

        assert registry.commands() == [foo]

    def test_overwrites_existing_command_with_same_name(self, make_fake_command: MakeFakeCommand) -> None:
        first = make_fake_command("foo", "first")
        second = make_fake_command("foo", "second")
        registry = CommandRegistry(commands=[first])

        registry.register_command(second)

        assert registry.commands() == [second]


class TestExecuteCommand:
    def test_calls_execute_with_given_arguments(self, make_fake_command: MakeFakeCommand) -> None:
        foo = make_fake_command("foo")
        registry = CommandRegistry(commands=[foo])

        registry.execute_command("foo", "a", "b")

        foo.mock_execute.assert_called_once_with("a", "b")

    def test_calls_execute_with_no_arguments(self, make_fake_command: MakeFakeCommand) -> None:
        foo = make_fake_command("foo")
        registry = CommandRegistry(commands=[foo])

        registry.execute_command("foo")

        foo.mock_execute.assert_called_once_with()

    def test_raises_unknown_command_error_for_unregistered_command(self) -> None:
        registry = CommandRegistry(commands=[])

        with pytest.raises(UnknownCommandError) as exc_info:
            registry.execute_command("missing")

        assert exc_info.value.command_name == "missing"


class TestCommands:
    def test_returns_list_of_registered_commands(self, make_fake_command: MakeFakeCommand) -> None:
        foo = make_fake_command("foo")
        registry = CommandRegistry(commands=[foo])

        assert registry.commands() == [foo]
