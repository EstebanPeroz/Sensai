from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete
from sqlalchemy.exc import ArgumentError, IntegrityError
from sqlalchemy.orm import Session

from sensai.llm.message import Message, Role
from sensai.memory.persistent_db.error import DbConnectionError, InvalidInstanceError
from sensai.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai.memory.persistent_db.sql_alchemy.table import BranchTable, MessageTable, PersonaTable


class FakeProviderManager:
    def __init__(self) -> None:
        self.models: list[str] = []

    def set_model(self, model: str) -> bool:
        self.models.append(model)
        return True


class FakeConversation:
    """Only the attributes `get_branch` touches: `uuid` and `history.replace`."""

    def __init__(self) -> None:
        self.uuid = None
        self.replaced: list[Message] | None = None
        self.history = self

    def replace(self, messages: list[Message]) -> None:
        self.replaced = messages


def fill(db: SqlAlchemy, branch_id: UUID, *contents: str) -> None:
    for i, content in enumerate(contents):
        db.add_message_to_branch(branch_id, Message(Role.USER if i % 2 == 0 else Role.ASSISTANT, content))


class TestInit:
    def test_creates_the_schema(self, db: SqlAlchemy) -> None:
        with Session(db._engine) as session:
            assert session.query(BranchTable).count() == 0
            assert session.query(MessageTable).count() == 0

    def test_creates_the_db_file(self, db: SqlAlchemy, tmp_path: Path)   -> None:  # noqa: ARG002
        assert (tmp_path / "test.db").exists()

    def test_data_survives_a_new_instance(self, db: SqlAlchemy) -> None:
        branch = db.create_branch()
        assert branch is not None
        assert SqlAlchemy().get_branch_list() == {"Branch": branch}

    def test_invalid_url_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)

        def boom(*_a: object, **_k: object) -> None:
            msg = "bad"
            raise ArgumentError(msg)

        monkeypatch.setattr("sensai.memory.persistent_db.sql_alchemy.core.create_engine", boom)
        with pytest.raises(DbConnectionError):
            SqlAlchemy()


class TestCreateBranch:
    def test_new_empty_branch(self, db: SqlAlchemy) -> None:
        branch = db.create_branch()

        assert branch is not None
        assert db.get_messages(branch) == []
        assert db.get_branch_list() == {"Branch": branch}

    def test_new_branches_get_unique_names(self, db: SqlAlchemy) -> None:
        ids = [db.create_branch() for _ in range(3)]

        assert len(set(ids)) == 3
        assert list(db.get_branch_list()) == ["Branch", "Branch 2", "Branch 3"]

    def test_copy_has_new_id_and_same_messages(self, db: SqlAlchemy) -> None:
        source = db.create_branch()
        fill(db, source, "a", "b", "c")

        copy = db.create_branch(source)

        assert copy is not None
        assert copy != source
        assert db.get_messages(copy) == db.get_messages(source)

    def test_copy_is_independent(self, db: SqlAlchemy) -> None:
        source = db.create_branch()
        fill(db, source, "a")
        copy = db.create_branch(source)

        fill(db, copy, "only in copy")
        db.remove_branch(source)

        assert [m.content for m in db.get_messages(copy)] == ["a", "only in copy"]

    def test_copy_names_do_not_collide(self, db: SqlAlchemy) -> None:
        source = db.create_branch()

        db.create_branch(source)
        db.create_branch(source)

        assert list(db.get_branch_list()) == ["Branch", "Branch 2", "Branch 3"]

    def test_copy_of_a_copy_gets_a_free_name(self, db: SqlAlchemy) -> None:
        source = db.create_branch()
        second = db.create_branch(source)

        db.create_branch(second)

        assert list(db.get_branch_list()) == ["Branch", "Branch 2", "Branch 2 2"]

    def test_copy_keeps_model_name(self, db: SqlAlchemy) -> None:
        source = db.create_branch()
        with Session(db._engine) as session:
            session.get(BranchTable, source).model_name = "llama"
            session.commit()

        copy = db.create_branch(source)

        with Session(db._engine) as session:
            assert session.get(BranchTable, copy).model_name == "llama"

    def test_copy_message_ids_are_new_and_linked_to_the_copy(self, db: SqlAlchemy) -> None:
        source = db.create_branch()
        fill(db, source, "a", "b")

        copy = db.create_branch(source)

        with Session(db._engine) as session:
            src_msgs = session.get(BranchTable, source).messages
            copy_msgs = session.get(BranchTable, copy).messages
            assert len(copy_msgs) == 2
            assert {m.id for m in copy_msgs}.isdisjoint({m.id for m in src_msgs})
            assert all(m.branch_id == copy for m in copy_msgs)
            assert session.query(MessageTable).count() == 4

    def test_unknown_source_returns_none(self, db: SqlAlchemy) -> None:
        assert db.create_branch(uuid4()) is None
        assert db.get_branch_list() == {}


class TestGetBranchList:
    def test_empty(self, db: SqlAlchemy) -> None:
        assert db.get_branch_list() == {}

    def test_maps_name_to_id(self, db: SqlAlchemy) -> None:
        first = db.create_branch()
        second = db.create_branch()

        assert db.get_branch_list() == {"Branch": first, "Branch 2": second}


class TestGetBranch:
    def test_loads_uuid_model_and_messages(self, db: SqlAlchemy) -> None:
        branch = db.create_branch()
        fill(db, branch, "hi", "hello")
        with Session(db._engine) as session:
            session.get(BranchTable, branch).model_name = "llama"
            session.commit()
        conv, manager = FakeConversation(), FakeProviderManager()

        db.get_branch(conv, branch, manager)

        assert conv.uuid == branch
        assert manager.models == ["llama"]
        assert conv.replaced == [Message(Role.USER, "hi"), Message(Role.ASSISTANT, "hello")]

    def test_unknown_branch_leaves_conversation_untouched(self, db: SqlAlchemy) -> None:
        conv, manager = FakeConversation(), FakeProviderManager()

        db.get_branch(conv, uuid4(), manager)

        assert conv.uuid is None
        assert conv.replaced is None
        assert manager.models == []


class TestRemoveBranch:
    def test_removes_only_that_branch(self, db: SqlAlchemy) -> None:
        keep = db.create_branch()
        drop = db.create_branch()

        db.remove_branch(drop)

        assert db.get_branch_list() == {"Branch": keep}

    def test_cascades_to_messages(self, db: SqlAlchemy) -> None:
        branch = db.create_branch()
        fill(db, branch, "a", "b")

        db.remove_branch(branch)

        with Session(db._engine) as session:
            assert session.query(MessageTable).count() == 0

    def test_does_not_touch_other_branch_messages(self, db: SqlAlchemy) -> None:
        keep = db.create_branch()
        fill(db, keep, "a")
        drop = db.create_branch(keep)

        db.remove_branch(drop)

        assert [m.content for m in db.get_messages(keep)] == ["a"]

    def test_unknown_branch_raises(self, db: SqlAlchemy) -> None:
        with pytest.raises(InvalidInstanceError):
            db.remove_branch(uuid4())


class TestAddMessageToBranch:
    def test_appends_in_order(self, db: SqlAlchemy) -> None:
        branch = db.create_branch()

        fill(db, branch, "1", "2", "3")

        assert [m.content for m in db.get_messages(branch)] == ["1", "2", "3"]

    def test_keeps_role(self, db: SqlAlchemy) -> None:
        branch = db.create_branch()
        for role in Role:
            db.add_message_to_branch(branch, Message(role, role.value))

        assert db.get_messages(branch) == [Message(role, role.value) for role in Role]

    def test_only_goes_to_target_branch(self, db: SqlAlchemy) -> None:
        a = db.create_branch()
        b = db.create_branch()

        fill(db, a, "x")

        assert db.get_messages(b) == []

    def test_unknown_branch_raises_and_stores_nothing(self, db: SqlAlchemy) -> None:
        with pytest.raises(InvalidInstanceError):
            db.add_message_to_branch(uuid4(), Message(Role.USER, "x"))

        with Session(db._engine) as session:
            assert session.query(MessageTable).count() == 0


class TestGetMessages:
    def test_empty_branch(self, db: SqlAlchemy) -> None:
        assert db.get_messages(db.create_branch()) == []

    def test_unknown_branch_returns_empty(self, db: SqlAlchemy) -> None:
        assert db.get_messages(uuid4()) == []

    def test_returns_message_objects_oldest_first(self, db: SqlAlchemy) -> None:
        branch = db.create_branch()
        fill(db, branch, "first", "second")

        assert db.get_messages(branch) == [Message(Role.USER, "first"), Message(Role.ASSISTANT, "second")]


def make_persona(db: SqlAlchemy, name: str = "p") -> UUID:
    with Session(db._engine) as session:
        persona = PersonaTable(name=name, description="d", prompt="x")
        session.add(persona)
        session.commit()
        return persona.id


class TestChangeBranchPersona:
    def test_sets_the_persona(self, db: SqlAlchemy) -> None:
        branch = db.create_branch()
        persona = make_persona(db)

        assert db.change_branch_persona(branch, persona) is True

        with Session(db._engine) as session:
            assert session.get(BranchTable, branch).persona_id == persona
            assert [b.id for b in session.get(PersonaTable, persona).branches] == [branch]

    def test_can_switch_persona(self, db: SqlAlchemy) -> None:
        branch = db.create_branch()
        first, second = make_persona(db, "a"), make_persona(db, "b")
        db.change_branch_persona(branch, first)

        assert db.change_branch_persona(branch, second) is True

        with Session(db._engine) as session:
            assert session.get(BranchTable, branch).persona_id == second
            assert session.get(PersonaTable, first).branches == []

    def test_unknown_persona_fails_and_changes_nothing(self, db: SqlAlchemy) -> None:
        branch = db.create_branch()
        persona = make_persona(db)
        db.change_branch_persona(branch, persona)

        assert db.change_branch_persona(branch, uuid4()) is False

        with Session(db._engine) as session:
            assert session.get(BranchTable, branch).persona_id == persona

    def test_unknown_branch_fails(self, db: SqlAlchemy) -> None:
        assert db.change_branch_persona(uuid4(), make_persona(db)) is False

    def test_copy_keeps_the_persona(self, db: SqlAlchemy) -> None:
        branch = db.create_branch()
        persona = make_persona(db)
        db.change_branch_persona(branch, persona)

        copy = db.create_branch(branch)

        with Session(db._engine) as session:
            assert session.get(BranchTable, copy).persona_id == persona


class TestForeignKeys:
    def test_message_with_unknown_branch_is_rejected(self, db: SqlAlchemy) -> None:
        with Session(db._engine) as session:
            session.add(MessageTable(branch_id=uuid4(), role=Role.USER, content="x"))
            with pytest.raises(IntegrityError):
                session.commit()

    def test_branch_with_unknown_persona_is_rejected(self, db: SqlAlchemy) -> None:
        with Session(db._engine) as session:
            session.add(BranchTable(name="b", model_name="", persona_id=uuid4()))
            with pytest.raises(IntegrityError):
                session.commit()

    def test_deleting_a_persona_unsets_it_on_branches(self, db: SqlAlchemy) -> None:
        branch = db.create_branch()
        persona = make_persona(db)
        db.change_branch_persona(branch, persona)

        with Session(db._engine) as session:
            session.delete(session.get(PersonaTable, persona))
            session.commit()

        with Session(db._engine) as session:
            assert session.get(BranchTable, branch).persona_id is None

    def test_database_cascades_branch_delete_to_messages(self, db: SqlAlchemy) -> None:
        branch = db.create_branch()
        fill(db, branch, "a", "b")

        with Session(db._engine) as session:
            session.execute(delete(BranchTable).where(BranchTable.id == branch))
            session.commit()
            assert session.query(MessageTable).count() == 0

    def test_every_connection_enforces_foreign_keys(self, db: SqlAlchemy) -> None:
        for _ in range(3):
            with db._engine.connect() as conn:
                assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
