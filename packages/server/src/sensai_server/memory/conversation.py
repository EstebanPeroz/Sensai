import asyncio
from typing import TYPE_CHECKING
from uuid import uuid4

from sensai_protocol.error import SensaiError

from sensai_server.memory.history import History
from sensai_server.memory.message import Message

if TYPE_CHECKING:
    from uuid import UUID

    from sensai_protocol.server_messages import Role

    from sensai_server.app.outbound import Outbound
    from sensai_server.llm.persona import Persona
    from sensai_server.memory.persistent_db import PersistentDatabase


class Conversation:
    """Drives a chat exchange with a provider, streaming its response back to a UI.

    Every call to the model, whatever its outcome, is emitted on `events` as an `Interaction`.

    `persona` is the persona applied to this conversation only. A new conversation has none. It is a snapshot,
    kept up to date by `SessionRegistry.refresh_persona`, and it is never stored in `history`: the chat puts its
    system message before the history when it builds the payload, so changing persona leaves no old prompt behind.

    `rag_id` is the id of the RAG of this conversation, which names its Qdrant collection. A new conversation gets
    one before its branch exists, so documents can be indexed before the first message. A branch loaded from the
    DB brings its own, or None when it has no RAG yet.
    """

    history: History
    model: str | None
    persona: Persona | None
    uuid: UUID | None = None
    rag_id: UUID | None
    _model_error: str = "Model is missing in current conversation"

    def __init__(self) -> None:
        """Init the conversation on the `pipeline` it chats through, the `events` it reports on, and its cache."""
        self.model = None
        self.persona = None
        self.history = History()
        self.rag_id = uuid4()

    def add_conv_to_db(self, db: PersistentDatabase, uuid: UUID | None = None, name: str = "Branch") -> None:
        """Init the conversation in the DB if first message.

        Raise SensaiError: If the current Conversation do not have model

        """
        if self.model is None:
            raise SensaiError(self._model_error)
        self.uuid = db.create_branch(model=self.model, current_branch=uuid, name=name)

    async def save_message(self, role: Role, content: str, out: Outbound, db: PersistentDatabase) -> bool:
        """Add message to History and db."""
        if not self.uuid or not db:
            await out.send_system_message("Failed to add message")
            return False

        try:
            uuid = await asyncio.shield(asyncio.to_thread(db.add_message_to_branch, self.uuid, role, content))
        except SensaiError as err:
            await out.send_system_message(err.message)
            return False

        if not uuid:
            await out.send_system_message("Failed to add message")
            return False

        self.history.insert(Message(uuid=uuid, role=role, content=content))
        return True
