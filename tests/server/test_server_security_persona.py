from pathlib import Path

import pytest
from sensai_server.config.settings import DEFAULT_SECURITY_PERSONA_FILE
from sensai_server.llm.persona import PersonaFileError, PersonaManager
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy

from tests.server.conftest import db_url

SECURITY_NAME = "Sentinelle de Sécurité"


@pytest.fixture
def db(tmp_path: Path) -> SqlAlchemy:
    return SqlAlchemy(db_url(tmp_path))


def write_persona(tmp_path: Path, name: str, prompt: str = "Be careful.") -> Path:
    path = tmp_path / f"{name}.toml"
    path.write_text(f'name = "{name}"\ndescription = "A persona."\nprompt = "{prompt}"\n', encoding="utf-8")
    return path


class TestSeeding:
    def test_starts_without_security_persona_by_default(self, db: SqlAlchemy) -> None:
        manager = PersonaManager(db)

        assert manager.security_personas() == []
        assert manager.get_personas() == []

    def test_seeds_the_packaged_security_persona(self, db: SqlAlchemy) -> None:
        manager = PersonaManager(db, [DEFAULT_SECURITY_PERSONA_FILE])

        [security] = manager.security_personas()
        assert security.name == SECURITY_NAME
        assert "Zero-Trust" in security.prompt
        assert manager.get_persona(SECURITY_NAME) == security

    def test_seeds_a_persona_file_from_the_disk(self, db: SqlAlchemy, tmp_path: Path) -> None:
        path = write_persona(tmp_path, "Guard")

        manager = PersonaManager(db, [str(path)])

        assert [persona.name for persona in manager.security_personas()] == ["Guard"]

    def test_keeps_the_order_of_the_settings(self, db: SqlAlchemy, tmp_path: Path) -> None:
        files = [str(write_persona(tmp_path, "Second")), DEFAULT_SECURITY_PERSONA_FILE]

        manager = PersonaManager(db, files)

        assert [persona.name for persona in manager.security_personas()] == ["Second", SECURITY_NAME]

    def test_reuses_the_stored_persona_on_restart(self, db: SqlAlchemy) -> None:
        first = PersonaManager(db, [DEFAULT_SECURITY_PERSONA_FILE]).security_personas()

        again = PersonaManager(db, [DEFAULT_SECURITY_PERSONA_FILE])

        assert again.security_personas() == first
        assert len(again.get_personas()) == 1

    def test_keeps_a_stored_persona_edited_since(self, db: SqlAlchemy) -> None:
        PersonaManager(db, [DEFAULT_SECURITY_PERSONA_FILE]).update_persona(SECURITY_NAME, prompt="Edited.")

        [security] = PersonaManager(db, [DEFAULT_SECURITY_PERSONA_FILE]).security_personas()

        assert security.prompt == "Edited."

    def test_returns_a_copy_of_the_security_personas(self, db: SqlAlchemy) -> None:
        manager = PersonaManager(db, [DEFAULT_SECURITY_PERSONA_FILE])

        manager.security_personas().clear()

        assert len(manager.security_personas()) == 1


class TestInvalidFiles:
    def test_missing_file_raises(self, db: SqlAlchemy, tmp_path: Path) -> None:
        with pytest.raises(PersonaFileError, match="cannot read it"):
            PersonaManager(db, [str(tmp_path / "missing.toml")])

    def test_unknown_package_raises(self, db: SqlAlchemy) -> None:
        with pytest.raises(PersonaFileError, match="no package 'not_a_package'"):
            PersonaManager(db, ["not_a_package/security.toml"])

    def test_missing_packaged_file_raises(self, db: SqlAlchemy) -> None:
        with pytest.raises(PersonaFileError, match="cannot read it"):
            PersonaManager(db, ["sensai_server.config.personas/missing.toml"])

    def test_invalid_persona_raises(self, db: SqlAlchemy, tmp_path: Path) -> None:
        path = tmp_path / "bad.toml"
        path.write_text('name = "Bad"\n', encoding="utf-8")

        with pytest.raises(PersonaFileError, match="missing key"):
            PersonaManager(db, [str(path)])

    def test_non_utf8_file_raises(self, db: SqlAlchemy, tmp_path: Path) -> None:
        path = tmp_path / "latin1.toml"
        path.write_bytes(b'name = "S\xe9curit\xe9"\n')

        with pytest.raises(PersonaFileError, match="not valid UTF-8"):
            PersonaManager(db, [str(path)])
