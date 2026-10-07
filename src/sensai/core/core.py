from typing import TYPE_CHECKING

from sensai import ui
from sensai.commands.help import Help
from sensai.commands.model import Model
from sensai.commands.registry import CommandRegistry, UnknownCommandError
from sensai.error import SensaiError
from sensai.history.conversation import Conversation
from sensai.llm.error import ModelError
from sensai.llm.message import Role
from sensai.llm.provider_manager import ProviderManager

if TYPE_CHECKING:
    import argparse

    from sensai.config.settings import Settings


class Core:
    """Wires a UI to a conversation, dispatching UI events until the user quits."""

    conversation: Conversation
    _provider_manager: ProviderManager
    _ui: ui.UIAdapter
    _command_registry: CommandRegistry
    _settings: Settings

    def __init__(self, ui: ui.UIAdapter, settings: Settings, args: argparse.Namespace) -> None:
        """Set up the provider manager and conversation. Safe to call before `ui.run()` starts."""
        self._ui = ui
        self._settings = settings

        self._provider_manager = ProviderManager(settings.llm)
        if not (self._provider_manager.set_model(args.model)):
            lines = [f"Model '{args.model}' not found in provider manager.", "Available models:"]
            lines.extend(f"- {model}" for model in sorted(self._provider_manager.get_models()))
            msg = "\n".join(lines)
            raise ModelError(msg)
        self.conversation = Conversation(self._provider_manager)
        self._command_registry = self._build_command_registry()

    def run(self) -> None:
        """Wait for `ui` to be ready, then dispatch its events to the conversation until quit.

        Meant to run off the main thread, since it blocks on `ui` being open, which
        requires `ui.run()` to be driving the UI's event loop on the main thread.
        """
        self._ui.open()

        self._ui.set_completions(
            {f"/{command.name}": command.completions() for command in self._command_registry.commands()}
        )

        event: ui.Event | None = None
        while not self._is_quit(event):
            self._ui.wait_event()
            event = self._ui.get_event()
            if event is None:
                continue
            if self._is_quit(event):
                break
            if event.type == ui.EventType.UserContent:
                self._launch_conv_chat(event.content)
            elif event.type == ui.EventType.Command:
                self._launch_command(event.content)

        self._ui.close()

    def _build_command_registry(self) -> CommandRegistry:
        """Build a command registry with all available commands."""
        registry = CommandRegistry([Model(self._ui, self._provider_manager)])
        registry.register_command(Help(self._ui, registry))
        return registry

    def _launch_command(self, content: str) -> None:
        """Parse `content` as a `/name args...` command and run it, reporting unknown commands to the UI."""
        parts = content.removeprefix("/").split()
        if not parts:
            return
        name, *args = parts
        try:
            self._ui.send_input(Role.USER, content, command=True)
            self._command_registry.execute_command(name, *args)
        except UnknownCommandError as err:
            self._ui.send_system_message(str(err))

    def _launch_conv_chat(self, content: str) -> None:
        """Send `content` through the conversation, reporting any provider error instead of raising."""
        if self.conversation.uuid is None:
            self.conversation.add_conv_to_db(None, content)
        try:
            self.conversation.chat(content, self._ui)
        except SensaiError as err:
            print(str(err))
            # replace with log

    @staticmethod
    def _is_quit(event: ui.Event | None) -> bool:
        return event is not None and event.type is ui.EventType.Command and event.content == "/q"
