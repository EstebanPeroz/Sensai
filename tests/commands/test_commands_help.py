from sensai.commands.help import Help
from sensai.commands.registry import CommandRegistry
from tests.commands.fakes import FakeUI, MakeFakeCommand


class TestHelp:
    def test_displays_header_and_registered_commands(self, fake_ui: FakeUI, make_fake_command: MakeFakeCommand) -> None:
        zebra = make_fake_command("zebra", "does zebra things")
        registry = CommandRegistry(commands=[zebra])
        help_command = Help(fake_ui, registry)
        registry.register_command(help_command)

        help_command.execute()

        assert len(fake_ui.messages) == 1
        message = fake_ui.messages[0]
        assert "Available commands:" in message
        assert "- zebra: does zebra things" in message
        assert "- help: Display available commands." in message

    def test_displays_commands_in_alphabetical_order(self, fake_ui: FakeUI, make_fake_command: MakeFakeCommand) -> None:
        zebra = make_fake_command("zebra")
        alpha = make_fake_command("alpha")
        registry = CommandRegistry(commands=[zebra, alpha])
        help_command = Help(fake_ui, registry)

        help_command.execute()

        lines = [line for line in fake_ui.messages[0].splitlines() if line.startswith("- ")]
        assert lines == ["- alpha: help text", "- zebra: help text"]

    def test_ignores_extra_arguments(self, fake_ui: FakeUI) -> None:
        help_command = Help(fake_ui, CommandRegistry(commands=[]))

        help_command.execute("unused", "args")

        assert fake_ui.messages == ["Available commands:"]
