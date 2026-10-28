from typing import TYPE_CHECKING

from sensai_server.memory import History

if TYPE_CHECKING:
    from uuid import UUID

    from sensai_server.llm.provider import ProviderManager


class Conversation:
    """Drives a chat exchange with a provider, streaming its response back to a UI.

    Every call to the model, whatever its outcome, is emitted on `events` as an `Interaction`.
    """

    _provider_manager: ProviderManager
    history: History
    uuid: UUID | None = None

    def __init__(
        self,
        provider_manager: ProviderManager,
    ) -> None:
        """Init the conversation on the `pipeline` it chats through, the `events` it reports on, and its cache."""
        self._provider_manager = provider_manager
        self.history = History()
