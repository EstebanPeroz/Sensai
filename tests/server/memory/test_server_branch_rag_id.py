from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sensai_server.llm.provider import ProviderManager
from sensai_server.memory import Conversation
from sensai_server.memory.persistent_db.error import DatabaseError, InvalidInstanceError
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy

from tests.server.conftest import MODEL, SETTINGS, db_url


@pytest.fixture
def db(tmp_path: Path) -> SqlAlchemy:
    return SqlAlchemy(db_url(tmp_path))


@pytest.fixture
def branch(db: SqlAlchemy) -> UUID:
    branch = db.create_branch(model=MODEL, name="Saved")
    assert branch is not None
    return branch


class TestBranchRagId:
    def test_a_new_branch_has_no_rag_id(self, db: SqlAlchemy, branch: UUID) -> None:
        assert db.get_branch_rag_id(branch) is None

    def test_a_rag_id_is_read_back(self, db: SqlAlchemy, branch: UUID) -> None:
        rag_id = uuid4()
        db.set_branch_rag_id(branch, rag_id)
        assert db.get_branch_rag_id(branch) == rag_id

    def test_a_rag_id_survives_a_reopen(self, tmp_path: Path, db: SqlAlchemy, branch: UUID) -> None:
        rag_id = uuid4()
        db.set_branch_rag_id(branch, rag_id)
        assert SqlAlchemy(db_url(tmp_path)).get_branch_rag_id(branch) == rag_id

    def test_a_copied_branch_has_no_rag_id(self, db: SqlAlchemy, branch: UUID) -> None:
        db.set_branch_rag_id(branch, uuid4())
        fork = db.create_branch(model=MODEL, current_branch=branch)
        assert fork is not None
        assert db.get_branch_rag_id(fork) is None

    def test_two_branches_cannot_share_a_rag_id(self, db: SqlAlchemy, branch: UUID) -> None:
        other = db.create_branch(model=MODEL, name="Other")
        assert other is not None
        rag_id = uuid4()
        db.set_branch_rag_id(branch, rag_id)
        with pytest.raises(DatabaseError):
            db.set_branch_rag_id(other, rag_id)
        assert db.get_branch_rag_id(other) is None

    def test_an_unknown_branch_raises(self, db: SqlAlchemy) -> None:
        with pytest.raises(InvalidInstanceError):
            db.get_branch_rag_id(uuid4())
        with pytest.raises(InvalidInstanceError):
            db.set_branch_rag_id(uuid4(), uuid4())

    def test_loading_a_branch_brings_its_rag_id(self, db: SqlAlchemy, branch: UUID) -> None:
        rag_id = uuid4()
        db.set_branch_rag_id(branch, rag_id)
        conv = Conversation()
        db.get_branch(conv, branch, ProviderManager(SETTINGS.llm))
        assert conv.rag_id == rag_id

    def test_loading_a_branch_without_rag_clears_the_rag_id(self, db: SqlAlchemy, branch: UUID) -> None:
        conv = Conversation()
        db.get_branch(conv, branch, ProviderManager(SETTINGS.llm))
        assert conv.rag_id is None

    def test_an_export_leaves_the_rag_id_out(self, db: SqlAlchemy, branch: UUID) -> None:
        db.set_branch_rag_id(branch, uuid4())
        export = db.export_branch(branch)
        assert export is not None
        assert "rag_id" not in export["branch"]
        db.import_branch(export)
        imported = next(uuid for uuid, name in db.get_branches().items() if uuid != branch)
        assert db.get_branch_rag_id(imported) is None
