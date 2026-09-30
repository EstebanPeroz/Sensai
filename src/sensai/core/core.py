import threading

from sensai import ui
from sensai.commands.help import Help
from sensai.commands.model import Model
from sensai.commands.registry import CommandRegistry, UnknownCommandError
from sensai.error import SensaiError
from sensai.history.conversation import Conversation
from sensai.llm.provider_manager import ProviderManager


def _is_quit(event: ui.Event | None) -> bool:
    return event is not None and event.type is ui.EventType.Command and event.content == "/q"


class Core:
    """Wires a UI to a conversation, dispatching UI events until the user quits."""

    conversation: Conversation
    _provider_manager: ProviderManager
    _ui: ui.UIAdapter
    _command_registry: CommandRegistry

    def __init__(self, ui: ui.UIAdapter) -> None:
        """Set up the provider manager and conversation, then wait for `ui` to be ready."""
        self._provider_manager = ProviderManager()
        self._ui = ui
        base = self._provider_manager.get_base_model()
        if base is not None:
            self.conversation = Conversation(base[1], base[0])
        self._command_registry = self._build_command_registry()
        while ui.open() is False:
            threading.Event().wait(0.1)

    def __del__(self) -> None:
        """Close the UI when the core is garbage-collected."""
        self._ui.close()

    def run(self) -> None:
        """Block, dispatching UI events to the conversation until a quit command is received."""
        event: ui.Event | None = None
        while not _is_quit(event):
            self._ui.wait_event()
            event = self._ui.get_event()
            if event is None:
                continue
            if _is_quit(event):
                break
            if event.type == ui.EventType.UserContent:
                self._launch_conv_chat(event.content)
            elif event.type == ui.EventType.Command:
                self._launch_command(event.content)

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
            self._command_registry.execute_command(name, *args)
        except UnknownCommandError as err:
            self._ui.send_system_message(str(err))

    def _launch_conv_chat(self, content: str) -> None:
        """Send `content` through the conversation, reporting any provider error instead of raising."""
        try:
            self.conversation.chat(content, self._ui)
        except SensaiError as err:
            # log + ui update from err
            print(err)
