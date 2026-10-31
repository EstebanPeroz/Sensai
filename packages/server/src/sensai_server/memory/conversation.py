from typing import TYPE_CHECKING

from sensai_protocol.error import SensaiError

from sensai_server.memory.history import History

if TYPE_CHECKING:
    from uuid import UUID

    from sensai_server.llm.persona import Persona
    from sensai_server.memory.persistent_db import PersistentDatabase


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
    _model_error: str = "Model is missing in current conversation"

    def __init__(self) -> None:
        """Init the conversation on the `pipeline` it chats through, the `events` it reports on, and its cache."""
        self.model = None
        self.persona = None
        self.history = History()

    def add_conv_to_db(self, db: PersistentDatabase, uuid: UUID | None = None, name: str = "Branch") -> None:
        """Init the conversation in the DB if first message.

        Raise SensaiError: If the current Conversation do not have model

        """
        if self.model is None:
            raise SensaiError(self._model_error)
        self.uuid = db.create_branch(uuid, name)
        if self.uuid:
            db.set_branch_model(self.uuid, self.model)
