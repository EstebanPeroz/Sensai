from typing import TYPE_CHECKING

from sensai_server.memory.history import History

if TYPE_CHECKING:
    from uuid import UUID


class Conversation:
    """Drives a chat exchange with a provider, streaming its response back to a UI.

    Every call to the model, whatever its outcome, is emitted on `events` as an `Interaction`.
    """

    history: History
    model: str | None
    uuid: UUID | None = None

    def __init__(self) -> None:
        """Init the conversation on the `pipeline` it chats through, the `events` it reports on, and its cache."""
        self.model = None
        self.history = History()
