from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from importlib import resources
from typing import TYPE_CHECKING

from sensai_server.llm.persona.error import DuplicatePersonaError, PersonaNotRegisteredError
from sensai_server.llm.persona.persona import DEFAULT_PERSONA_FILE, Persona
from sensai_server.memory.persistent_db.error import DatabaseError

if TYPE_CHECKING:
    from uuid import UUID

    from sensai_server.memory.persistent_db.adapter import PersistentDatabase


class PersonaManager:
    """Name-based access to the personas held by the database.

    Shared by every session: it holds no per-client state. The persona applied to a
    conversation is kept by that conversation.
    """

    _db: PersistentDatabase
    _store_error_message: str = "the persona could not be stored"
    _record_error_message: str = "the database returned a persona without an id"

    def __init__(self, db: PersistentDatabase) -> None:
        """Read and write the personas through `db`."""
        self._db = db

    def create_persona(self, draft: Persona) -> Persona:
        """Store the unsaved persona `draft` and return the stored one."""
        if self._find(draft.name) is not None:
            raise DuplicatePersonaError(draft.name)
        created = self._db.create_persona(draft.name, draft.description, datetime.now(UTC), draft.prompt)
        if created is None:
            raise DatabaseError(self._store_error_message)
        return created

    def create_default_persona(self) -> Persona:
        """Store the persona packaged with the server and return it, to seed a fresh database."""
        source = resources.files("sensai_server.config").joinpath(DEFAULT_PERSONA_FILE)
        draft = Persona.from_toml(source.read_text(encoding="utf-8"), DEFAULT_PERSONA_FILE)
        return self.create_persona(draft)

    def get_personas(self) -> list[Persona]:
        """Return every stored persona."""
        return self._db.get_personas()

    def get_persona(self, name: str) -> Persona:
        """Return the persona named `name`, whatever its case."""
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
        """Update the persona named `target_name`, leaving out the fields left to `None`, and return it."""
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
        return self._db.get_persona(uuid)

    def update_persona_from_draft(self, draft: Persona) -> Persona:
        """Update the stored persona named like the unsaved persona `draft` with its description and prompt."""
        return self.update_persona(draft.name, description=draft.description, prompt=draft.prompt)

    def _find(self, name: str) -> Persona | None:
        """Return the stored persona named `name`, whatever its case, or None if there is none."""
        key = name.casefold()
        return next((persona for persona in self.get_personas() if persona.name.casefold() == key), None)

    def _stored_uuid(self, persona: Persona) -> UUID:
        """Return the id of a persona read from the database, which always has one."""
        if persona.uuid is None:
            raise DatabaseError(self._record_error_message)
        return persona.uuid
