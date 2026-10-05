from typing import TYPE_CHECKING

from sensai import ui
from sensai.error import SensaiError
from sensai.history.conversation import Conversation
from sensai.llm.provider_manager import ProviderManager

if TYPE_CHECKING:
    from sensai.config.settings import Settings


class Core:
    """Wires a UI to a conversation, dispatching UI events until the user quits."""

    conversation: Conversation | None = None
    _provider_manager: ProviderManager
    _ui: ui.UIAdapter
    _settings: Settings

    def __init__(self, ui: ui.UIAdapter, settings: Settings) -> None:
        """Set up the provider manager and conversation. Safe to call before `ui.run()` starts."""
        self._ui = ui
        self._settings = settings

        self._provider_manager = ProviderManager(settings.llm)

    def run(self) -> None:
        """Wait for `ui` to be ready, then dispatch its events to the conversation until quit.

        Meant to run off the main thread, since it blocks on `ui` being open, which
        requires `ui.run()` to be driving the UI's event loop on the main thread.
        """
        self._ui.open()

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

        self._ui.close()

    def _launch_conv_chat(self, content: str) -> None:
        """Send `content` through the conversation, reporting any provider error instead of raising."""
        if self.conversation is None:
            self._ui.send_user_input("No conversation setted")
            # replace with log + ui error message
            return
        try:
            self.conversation.chat(content, self._ui)
        except SensaiError as err:
            print(str(err))
            # replace with log

    def init_conversation(self, model: str | None) -> None:
        """Set Conversation with given model."""
        if model is not None:
            provider = self._provider_manager.get_provider(model)
            if provider is not None:
                self.conversation = Conversation(provider, model)

    @staticmethod
    def _is_quit(event: ui.Event | None) -> bool:
        return event is not None and event.type is ui.EventType.Command and event.content == "/q"
