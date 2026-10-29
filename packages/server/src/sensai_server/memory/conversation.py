from typing import TYPE_CHECKING

from sensai_server.memory.history import History

if TYPE_CHECKING:
    from uuid import UUID

    from sensai_server.llm.persona import Persona


class Conversation:
    """Drives a chat exchange with a provider, streaming its response back to a UI.

    Every call to the model, whatever its outcome, is emitted on `events` as an `Interaction`.

    `persona` is the persona applied to this conversation only. A new conversation has none. It is a snapshot,
    kept up to date by `SessionRegistry.refresh_persona`, and it is never stored in `history`: the chat puts its
    system message before the history when it builds the payload, so changing persona leaves no old prompt behind.
    """

    history: History
    model: str | None
    persona: Persona | None
    uuid: UUID | None = None

    def __init__(self) -> None:
        """Init the conversation on the `pipeline` it chats through, the `events` it reports on, and its cache."""
        self.model = None
        self.persona = None
        self.history = History()
