from pathlib import Path
from uuid import UUID

import pytest
from sensai_server.llm.persona import Persona, PersonaManager
from sensai_server.llm.provider import ProviderManager
from sensai_server.memory import Conversation
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy

from tests.server.conftest import MODEL, SETTINGS, db_url


@pytest.fixture
def db(tmp_path: Path) -> SqlAlchemy:
    return SqlAlchemy(db_url(tmp_path))


@pytest.fixture
def persona(db: SqlAlchemy) -> Persona:
    return PersonaManager(db).create_default_persona()


def _branch(db: SqlAlchemy) -> UUID:
    branch = db.create_branch()
    assert branch is not None
    db.set_branch_model(branch, MODEL)
    return branch


class TestBranchPersona:
    def test_get_branch_restores_the_persona(self, db: SqlAlchemy, persona: Persona) -> None:
        branch = _branch(db)
        assert persona.uuid is not None
        assert db.set_branch_persona(branch, persona.uuid)
        conv = Conversation()
        db.get_branch(conv, branch, ProviderManager(SETTINGS.llm))
        assert conv.persona == persona

    def test_get_branch_without_persona_clears_it(self, db: SqlAlchemy, persona: Persona) -> None:
        branch = _branch(db)
        conv = Conversation()
        conv.persona = persona
        db.get_branch(conv, branch, ProviderManager(SETTINGS.llm))
        assert conv.persona is None

    def test_create_branch_keeps_the_persona(self, db: SqlAlchemy, persona: Persona) -> None:
        branch = _branch(db)
        assert persona.uuid is not None
        db.set_branch_persona(branch, persona.uuid)
        copy = db.create_branch(branch)
        assert copy is not None
        conv = Conversation()
        db.get_branch(conv, copy, ProviderManager(SETTINGS.llm))
        assert conv.persona == persona
