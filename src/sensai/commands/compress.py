from typing import TYPE_CHECKING, override

from sensai.commands.base import Command
from sensai.error import SensaiError

if TYPE_CHECKING:
    from sensai.history.conversation import Conversation
    from sensai.pipeline.modules.compression import Compressor


class Compress(Command):
    """Compress command class to summarise the oldest messages of the conversation without waiting for the budget."""

    name = "compress"
    help = "Summarise the oldest messages of the conversation now, to free some of its context."
    usage = "/compress"
    _conversation: Conversation
    _compressor: Compressor

    def __init__(self, conversation: Conversation, compressor: Compressor) -> None:
        """Initialize the Compress command with the conversation it shortens and the compressor that summarises."""
        super().__init__()
        self._conversation = conversation
        self._compressor = compressor

    @override
    def execute(self, *args: str) -> None:
        """Summarise the oldest messages of the current branch and tell how much it saved."""
        if args:
            self._ui.send_system_message(f"This command takes no argument. Usage: {self.usage}")
            return
        branch = self._conversation.uuid
        if branch is None:
            self._ui.send_system_message("Nothing to compress: the conversation has not started.")
            return
        try:
            compression = self._compressor.compress(branch, self._conversation.history.to_json())
        except SensaiError as err:
            self._ui.send_system_message(f"The conversation could not be compressed: {err.message}")
            return
        if compression is None:
            self._ui.send_system_message("Nothing to compress: only the last messages are left to summarise.")
            return
        self._ui.send_system_message(compression.describe())
