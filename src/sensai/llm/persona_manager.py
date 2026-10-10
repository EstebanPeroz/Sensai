from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sensai.llm.persona import DuplicatePersonaError, Persona, PersonaNotRegisteredError
from sensai.memory.persistent_db.error import DatabaseError

if TYPE_CHECKING:
    from uuid import UUID

    from sensai.memory.persistent_db.adapter import PersistentDatabase


class PersonaManager:
    """Name-based access to the personas held by the database, and to the one currently applied.

    The database owns the personas and addresses them by `UUID`; this manager is the
    user-facing side, where a persona is picked by name, and it remembers which one is
    applied to the session. A persona is validated before it reaches the database, so
    an empty field never gets stored.
    """

    _db: PersistentDatabase
    _active_persona: Persona | None
    _store_error_message: str = "the persona could not be stored"
    _record_error_message: str = "the database returned a persona without an id"

    def __init__(self, db: PersistentDatabase) -> None:
        """Read the personas from `db`, and start with none applied."""
        self._db = db
        self._active_persona = None

    def create_persona(self, name: str, description: str, prompt: str) -> Persona:
        """Store a new persona and return it.

        Raises:
            InvalidPersonaError: If a field is missing or empty.
            DuplicatePersonaError: If a persona is already named `name`.
            DatabaseError: If the persona could not be stored.

        """
        draft = Persona(uuid=None, name=name, description=description, prompt=prompt)
        if self._find(draft.name) is not None:
            raise DuplicatePersonaError(draft.name)
        created = self._db.create_persona(draft.name, draft.description, datetime.now(UTC), draft.prompt)
        if created is None:
            raise DatabaseError(self._store_error_message)
        return created

    def get_personas(self) -> list[Persona]:
        """Return every stored persona."""
        return self._db.get_personas()

    def get_persona(self, name: str) -> Persona:
        """Return the persona named `name`, whatever its case.

        Raises:
            PersonaNotRegisteredError: If no persona is named `name`.

        """
        persona = self._find(name)
        if persona is None:
            raise PersonaNotRegisteredError(name, [stored.name for stored in self.get_personas()])
        return persona

    def update_persona(
        self,
        target_name: str,
        *,
        new_name: str | None = None,
        description: str | None = None,
        prompt: str | None = None,
    ) -> Persona:
        """Update the persona named `target_name`, leaving out the fields left to `None`, and return it.

        Raises:
            PersonaNotRegisteredError: If no persona is named `target_name`.
            InvalidPersonaError: If a field is given but empty.
            DuplicatePersonaError: If `new_name` is already used by another persona.
            DatabaseError: If the update could not be stored.

        """
        target = self.get_persona(target_name)
        draft = replace(
            target,
            name=target.name if new_name is None else new_name,
            description=target.description if description is None else description,
            prompt=target.prompt if prompt is None else prompt,
        )
        clash = self._find(draft.name)
        if clash is not None and clash.uuid != target.uuid:
            raise DuplicatePersonaError(draft.name)
        uuid = self._stored_uuid(target)
        self._db.update_persona(uuid, name=new_name, description=description, prompt=prompt)
        updated = self._db.get_persona(uuid)
        if self._active_persona is not None and self._active_persona.uuid == uuid:
            self._active_persona = updated
        return updated

    def activate(self, name: str) -> Persona:
        """Apply the persona named `name` and return it.

        Raises:
            PersonaNotRegisteredError: If no persona is named `name`.

        """
        persona = self.get_persona(name)
        self._active_persona = persona
        return persona

    def deactivate(self) -> None:
        """Stop applying any persona."""
        self._active_persona = None

    def active_persona(self) -> Persona | None:
        """Return the persona currently applied, or None if there is none."""
        return self._active_persona

    def _find(self, name: str) -> Persona | None:
        """Return the stored persona named `name`, whatever its case, or None if there is none."""
        key = name.casefold()
        return next((persona for persona in self.get_personas() if persona.name.casefold() == key), None)

    def _stored_uuid(self, persona: Persona) -> UUID:
        """Return the id of a persona read from the database, which always has one.

        Raises:
            DatabaseError: If the record came back without an id.

        """
        if persona.uuid is None:
            raise DatabaseError(self._record_error_message)
        return persona.uuid
