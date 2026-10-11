from typing import TYPE_CHECKING, override

from sqlalchemy import Connection, Engine, create_engine, event, select
from sqlalchemy.exc import ArgumentError, SQLAlchemyError
from sqlalchemy.orm import Session

from sensai.llm.message import Message, Role
from sensai.llm.persona import Persona
from sensai.memory.persistent_db.adapter import PersistentDatabase
from sensai.memory.persistent_db.error import DbConnectionError, InvalidInstanceError
from sensai.memory.persistent_db.sql_alchemy.table import Base, BranchTable, MessageTable, PersonaTable

if TYPE_CHECKING:
    import sqlite3
    from datetime import datetime
    from uuid import UUID

    from sensai.history.conversation import Conversation
    from sensai.llm.provider_manager import ProviderManager


class SqlAlchemy(PersistentDatabase):
    """`PersistentDatabase` backed by SQLAlchemy (SQLite foreign keys are enforced)."""

    _engine: Engine
    _conn: Connection
    _uuid_error_essage: str = "given uuid is invalid"

    def __init__(self, url: str) -> None:
        """Connect to the database at `url` and create the missing tables.

        Raises:
            DbConnectionError: If `url` is not a valid database url.

        """
        super().__init__()

        try:
            self._engine = create_engine(url)
        except ArgumentError:
            msg = "Invalid url to database"
            raise DbConnectionError(msg) from None
        event.listen(self._engine, "connect", self._enable_foreign_keys)
        self._conn = self._engine.connect()
        Base.metadata.create_all(self._engine)

    @staticmethod
    def _enable_foreign_keys(dbapi_conn: sqlite3.Connection, _: object) -> None:
        """Make SQLite enforce foreign keys, which it ignores by default (per connection)."""
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    @override
    def create_branch(self, current_branch: UUID | None = None, name: str = "Branch") -> UUID | None:
        """Create a new branch and return its id if success or None if fail.

        If `current_branch` is given, the new branch is a deep copy of it
        (messages and tool calls included); otherwise an empty branch is made.
        """
        with Session(self._engine) as session:
            if current_branch is None:
                branch = BranchTable(name=self._free_name(session, name), model_name="")
            else:
                source = session.get(BranchTable, current_branch)
                if source is None:
                    return None
                branch = source.copy(self._free_name(session, source.name))
            session.add(branch)
            try:
                session.commit()
            except SQLAlchemyError:
                return None
            return branch.id

    @staticmethod
    def _free_name(session: Session, base: str) -> str:
        """Return `base` if unused, else `base 2`, `base 3`, ... (names are unique)."""
        taken = set(session.scalars(select(BranchTable.name).where(BranchTable.name.like(f"{base}%"))))
        if base not in taken:
            return base
        n = 2
        while f"{base} {n}" in taken:
            n += 1
        return f"{base} {n}"

    @override
    def get_branch_list(self) -> dict[str, UUID]:
        """Return the dict of all branches with name and uuid."""
        branches: dict[str, UUID] = {}
        with Session(self._engine) as session:
            source = session.scalars(select(BranchTable)).all()
            for branch in source:
                branches[branch.name] = branch.id
        return branches

    @override
    def get_branch(self, current: Conversation, target_branch: UUID, provider_manager: ProviderManager) -> None:
        """Replace the uuid, model and history of `current` with those of the target branch.

        Does nothing if no branch has this uuid.
        """
        with Session(self._engine) as session:
            source = session.get(BranchTable, target_branch)
            if source is None:
                return
            current.uuid = source.id
            provider_manager.set_model(source.model_name)
            current.history.replace(self._messages_from_branch(source))

    @override
    def remove_branch(self, uuid: UUID) -> None:
        """Remove the branch related to this uuid.

        Raises:
            InvalidInstanceError: If no branch has this uuid.

        """
        with Session(self._engine) as session:
            branch = session.get(BranchTable, uuid)
            if branch is None:
                raise InvalidInstanceError(self._uuid_error_essage)
            session.delete(branch)
            session.commit()

    @override
    def add_message_to_branch(self, branch_id: UUID, role: Role, content: str) -> UUID | None:
        """Append a message made of `role` and `content` at the end of the branch `branch_id`.

        Return the id of the new message, or None if it could not be saved.

        Raises:
            InvalidInstanceError: If no branch has this uuid.

        """
        with Session(self._engine) as session:
            branch = session.get(BranchTable, branch_id)
            if branch is None:
                raise InvalidInstanceError(self._uuid_error_essage)
            msg = MessageTable(branch_id=branch_id, role=role, content=content)
            session.add(msg)
            try:
                session.commit()
            except SQLAlchemyError:
                return None
            return msg.id

    @override
    def set_branch_model(self, branch: UUID, model: str) -> None:
        """Set the model used by the branch `branch` to `model`.

        Raises:
            InvalidInstanceError: If no branch has this uuid.

        """
        with Session(self._engine) as session:
            source = session.get(BranchTable, branch)
            if source is None:
                raise InvalidInstanceError(self._uuid_error_essage)
            source.model_name = model
            session.commit()

    @override
    def set_branch_persona(self, branch: UUID, persona: UUID) -> bool:
        """Set the persona used by the branch `branch` to `persona`.

        Return True on success, False if the branch or the persona does not exist
        or the change could not be saved.

        """
        with Session(self._engine) as session:
            source = session.get(BranchTable, branch)
            target = session.get(PersonaTable, persona)
            if source is None or target is None:
                return False
            source.persona = target
            try:
                session.commit()
            except SQLAlchemyError:
                return False
        return True

    @override
    def get_messages(self, branch: UUID) -> list[Message]:
        """Return every message of the branch `branch`, oldest first."""
        with Session(self._engine) as session:
            source = session.get(BranchTable, branch)
            if source is None:
                return []
            return self._messages_from_branch(source)

    @staticmethod
    def _messages_from_branch(branch: BranchTable) -> list[Message]:
        """Convert the stored messages of a branch into Message objects."""
        return [Message(msg.id, msg.role, msg.content) for msg in branch.messages]

    @override
    def create_persona(self, name: str, description: str, timestamp: datetime, prompt: str) -> Persona | None:
        """Create a persona and return its Class Version after adding it to the db.

        Not implemented yet: always returns None.

        Args:
            name: Display name of the persona.
            description: Short summary of what the persona is for.
            timestamp: Creation time of the persona.
            prompt: System prompt applied when the persona is active.

        """
        return None

    @override
    def get_personas(self) -> list[Persona]:
        """Return all personas (only the name is filled for now)."""
        personas: list[Persona] = []
        with Session(self._engine) as session:
            source = session.scalars(select(PersonaTable)).all()
            personas.extend(
                Persona(persona.name, "") for persona in source
            )  ## Update to use the next persona dataclass
        return personas

    @override
    def get_persona(self, persona: UUID) -> Persona:
        """Return the persona `persona` (placeholder values for now).

        Raises:
            InvalidInstanceError: If no persona has this id.

        """
        with Session(self._engine) as session:
            source = session.get(PersonaTable, persona)
            if source is None:
                raise InvalidInstanceError(self._uuid_error_essage)
            return Persona("...", "...")  ## Update to use the next persona dataclass

    @override
    def update_persona(
        self,
        persona: UUID,
        *,
        name: str | None = None,
        description: str | None = None,
        prompt: str | None = None,
    ) -> None:
        """Update the persona `persona`; fields left to `None` are unchanged.

        Not implemented yet: does nothing.

        Raises:
            InvalidInstanceError: If no persona has this id.

        """
