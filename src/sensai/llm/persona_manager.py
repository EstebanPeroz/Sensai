from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from sensai.llm.persona import (
    DEFAULT_PERSONA_FILE,
    DuplicatePersonaError,
    Persona,
    PersonaFileError,
    PersonaNotRegisteredError,
)
from sensai.memory.persistent_db.error import DatabaseError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from uuid import UUID

    from sensai.memory.persistent_db.adapter import PersistentDatabase


class PersonaManager:
    """Name-based access to the personas held by the database, and to the one currently applied."""

    _db: PersistentDatabase
    _active_persona: Persona | None
    _security_personas: list[Persona]
    _store_error_message: str = "the persona could not be stored"
    _record_error_message: str = "the database returned a persona without an id"

    def __init__(self, db: PersistentDatabase, security_files: Iterable[str] = ()) -> None:
        """Read the personas from `db`, seed the security ones listed in `security_files`, and apply none."""
        self._db = db
        self._security_personas = [self._seed(file) for file in security_files]
        self._active_persona = None

    def security_personas(self) -> list[Persona]:
        """Return the stored security personas, in the order the settings list them."""
        return list(self._security_personas)

    def _seed(self, file: str) -> Persona:
        """Store the persona described by `file`, or return the stored one that already goes by its name."""
        with _resolve(file) as path:
            existing = self._find(Persona.from_file(path).name)
            if existing is not None:
                return existing
            return self.create_persona_from_file(path)

    def create_persona_from_file(self, path: str | Path) -> Persona:
        """Store the persona described by the file at `path` and return  it."""
        draft = Persona.from_file(path)
        if self._find(draft.name) is not None:
            raise DuplicatePersonaError(draft.name)
        created = self._db.create_persona(draft.name, draft.description, datetime.now(UTC), draft.prompt)
        if created is None:
            raise DatabaseError(self._store_error_message)
        return created

    def create_default_persona(self) -> Persona:
        """Store the persona packaged with sensai and return it, to seed a fresh database."""
        source = resources.files("sensai.config").joinpath(DEFAULT_PERSONA_FILE)
        with resources.as_file(source) as path:
            return self.create_persona_from_file(path)

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
        updated = self._db.get_persona(uuid)
        if self._active_persona is not None and self._active_persona.uuid == uuid:
            self._active_persona = updated
        return updated

    def activate(self, name: str) -> Persona:
        """Apply the persona named `name` and return it."""
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
        """Return the id of a persona read from the database, which always has one."""
        if persona.uuid is None:
            raise DatabaseError(self._record_error_message)
        return persona.uuid


@contextmanager
def _resolve(file: str) -> Iterator[Path]:
    """Yield the path of `file`, read from the filesystem or, failing that, from the `<package>/<filename>` it names."""
    path = Path(file)
    if path.exists() or "/" not in file:
        yield path
        return
    package, _, filename = file.rpartition("/")
    try:
        source = resources.files(package).joinpath(filename)
    except (ImportError, TypeError) as err:
        raise PersonaFileError(path, f"no package '{package}' to read it from ({err})") from err
    with resources.as_file(source) as resolved:
        yield resolved
