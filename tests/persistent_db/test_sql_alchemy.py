from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete
from sqlalchemy.exc import ArgumentError, IntegrityError
from sqlalchemy.orm import Session

from sensai.history.conversation import Conversation
from sensai.llm.message import Message, Role
from sensai.llm.provider_manager import ProviderManager
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


def new_branch(db: SqlAlchemy, source: UUID | None = None) -> UUID:
    branch = db.create_branch(source)
    assert branch is not None
    return branch


def row[T](session: Session, table: type[T], pk: UUID) -> T:
    instance = session.get(table, pk)
    assert instance is not None
    return instance


def fill(db: SqlAlchemy, branch_id: UUID, *contents: str) -> None:
    for i, content in enumerate(contents):
        db.add_message_to_branch(branch_id, Role.USER if i % 2 == 0 else Role.ASSISTANT, content)


def turns(messages: list[Message]) -> list[tuple[Role, str]]:
    """Messages without their (per-row) uuid, for comparisons across branches."""
    return [(m.role, m.content) for m in messages]


class TestInit:
    def test_creates_the_schema(self, db: SqlAlchemy) -> None:
        with Session(db._engine) as session:
            assert session.query(BranchTable).count() == 0
            assert session.query(MessageTable).count() == 0

    def test_creates_the_db_file(self, db: SqlAlchemy, tmp_path: Path) -> None:  # noqa: ARG002
        assert (tmp_path / "test.db").exists()

    def test_data_survives_a_new_instance(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)
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
        branch = new_branch(db)

        assert branch is not None
        assert db.get_messages(branch) == []
        assert db.get_branch_list() == {"Branch": branch}

    def test_new_branches_get_unique_names(self, db: SqlAlchemy) -> None:
        ids = [new_branch(db) for _ in range(3)]

        assert len(set(ids)) == 3
        assert list(db.get_branch_list()) == ["Branch", "Branch 2", "Branch 3"]

    def test_copy_has_new_id_and_same_messages(self, db: SqlAlchemy) -> None:
        source = new_branch(db)
        fill(db, source, "a", "b", "c")

        copy = new_branch(db, source)

        assert copy is not None
        assert copy != source
        assert turns(db.get_messages(copy)) == turns(db.get_messages(source))

    def test_copy_is_independent(self, db: SqlAlchemy) -> None:
        source = new_branch(db)
        fill(db, source, "a")
        copy = new_branch(db, source)

        fill(db, copy, "only in copy")
        db.remove_branch(source)

        assert [m.content for m in db.get_messages(copy)] == ["a", "only in copy"]

    def test_copy_names_do_not_collide(self, db: SqlAlchemy) -> None:
        source = new_branch(db)

        new_branch(db, source)
        new_branch(db, source)

        assert list(db.get_branch_list()) == ["Branch", "Branch 2", "Branch 3"]

    def test_copy_of_a_copy_gets_a_free_name(self, db: SqlAlchemy) -> None:
        source = new_branch(db)
        second = new_branch(db, source)

        new_branch(db, second)

        assert list(db.get_branch_list()) == ["Branch", "Branch 2", "Branch 2 2"]

    def test_copy_keeps_model_name(self, db: SqlAlchemy) -> None:
        source = new_branch(db)
        with Session(db._engine) as session:
            row(session, BranchTable, source).model_name = "llama"
            session.commit()

        copy = new_branch(db, source)

        with Session(db._engine) as session:
            assert row(session, BranchTable, copy).model_name == "llama"

    def test_copy_message_ids_are_new_and_linked_to_the_copy(self, db: SqlAlchemy) -> None:
        source = new_branch(db)
        fill(db, source, "a", "b")

        copy = new_branch(db, source)

        with Session(db._engine) as session:
            src_msgs = row(session, BranchTable, source).messages
            copy_msgs = row(session, BranchTable, copy).messages
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
        first = new_branch(db)
        second = new_branch(db)

        assert db.get_branch_list() == {"Branch": first, "Branch 2": second}


class TestGetBranch:
    def test_loads_uuid_model_and_messages(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)
        fill(db, branch, "hi", "hello")
        with Session(db._engine) as session:
            row(session, BranchTable, branch).model_name = "llama"
            session.commit()
        conv, manager = FakeConversation(), FakeProviderManager()

        db.get_branch(cast(Conversation, conv), branch, cast(ProviderManager, manager))

        assert conv.uuid == branch
        assert manager.models == ["llama"]
        assert conv.replaced is not None
        assert turns(conv.replaced) == [(Role.USER, "hi"), (Role.ASSISTANT, "hello")]

    def test_unknown_branch_leaves_conversation_untouched(self, db: SqlAlchemy) -> None:
        conv, manager = FakeConversation(), FakeProviderManager()

        db.get_branch(cast(Conversation, conv), uuid4(), cast(ProviderManager, manager))

        assert conv.uuid is None
        assert conv.replaced is None
        assert manager.models == []


class TestRemoveBranch:
    def test_removes_only_that_branch(self, db: SqlAlchemy) -> None:
        keep = new_branch(db)
        drop = new_branch(db)

        db.remove_branch(drop)

        assert db.get_branch_list() == {"Branch": keep}

    def test_cascades_to_messages(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)
        fill(db, branch, "a", "b")

        db.remove_branch(branch)

        with Session(db._engine) as session:
            assert session.query(MessageTable).count() == 0

    def test_does_not_touch_other_branch_messages(self, db: SqlAlchemy) -> None:
        keep = new_branch(db)
        fill(db, keep, "a")
        drop = new_branch(db, keep)

        db.remove_branch(drop)

        assert [m.content for m in db.get_messages(keep)] == ["a"]

    def test_unknown_branch_raises(self, db: SqlAlchemy) -> None:
        with pytest.raises(InvalidInstanceError):
            db.remove_branch(uuid4())


class TestAddMessageToBranch:
    def test_appends_in_order(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)

        fill(db, branch, "1", "2", "3")

        assert [m.content for m in db.get_messages(branch)] == ["1", "2", "3"]

    def test_keeps_role(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)
        for role in Role:
            db.add_message_to_branch(branch, role, role.value)

        assert turns(db.get_messages(branch)) == [(role, role.value) for role in Role]

    def test_only_goes_to_target_branch(self, db: SqlAlchemy) -> None:
        a = new_branch(db)
        b = new_branch(db)

        fill(db, a, "x")

        assert db.get_messages(b) == []

    def test_unknown_branch_raises_and_stores_nothing(self, db: SqlAlchemy) -> None:
        with pytest.raises(InvalidInstanceError):
            db.add_message_to_branch(uuid4(), Role.USER, "x")

        with Session(db._engine) as session:
            assert session.query(MessageTable).count() == 0


class TestGetMessages:
    def test_empty_branch(self, db: SqlAlchemy) -> None:
        assert db.get_messages(new_branch(db)) == []

    def test_unknown_branch_returns_empty(self, db: SqlAlchemy) -> None:
        assert db.get_messages(uuid4()) == []

    def test_returns_message_objects_oldest_first(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)
        fill(db, branch, "first", "second")

        messages = db.get_messages(branch)
        assert turns(messages) == [(Role.USER, "first"), (Role.ASSISTANT, "second")]
        assert all(isinstance(m, Message) and m.uuid is not None for m in messages)


def make_persona(db: SqlAlchemy, name: str = "p") -> UUID:
    with Session(db._engine) as session:
        persona = PersonaTable(name=name, description="d", prompt="x")
        session.add(persona)
        session.commit()
        return persona.id


class TestSetBranchPersona:
    def test_sets_the_persona(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)
        persona = make_persona(db)

        assert db.set_branch_persona(branch, persona) is True

        with Session(db._engine) as session:
            assert row(session, BranchTable, branch).persona_id == persona
            assert [b.id for b in row(session, PersonaTable, persona).branches] == [branch]

    def test_can_switch_persona(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)
        first, second = make_persona(db, "a"), make_persona(db, "b")
        db.set_branch_persona(branch, first)

        assert db.set_branch_persona(branch, second) is True

        with Session(db._engine) as session:
            assert row(session, BranchTable, branch).persona_id == second
            assert row(session, PersonaTable, first).branches == []

    def test_unknown_persona_fails_and_changes_nothing(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)
        persona = make_persona(db)
        db.set_branch_persona(branch, persona)

        assert db.set_branch_persona(branch, uuid4()) is False

        with Session(db._engine) as session:
            assert row(session, BranchTable, branch).persona_id == persona

    def test_unknown_branch_fails(self, db: SqlAlchemy) -> None:
        assert db.set_branch_persona(uuid4(), make_persona(db)) is False

    def test_copy_keeps_the_persona(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)
        persona = make_persona(db)
        db.set_branch_persona(branch, persona)

        copy = new_branch(db, branch)

        with Session(db._engine) as session:
            assert row(session, BranchTable, copy).persona_id == persona


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
        branch = new_branch(db)
        persona = make_persona(db)
        db.set_branch_persona(branch, persona)

        with Session(db._engine) as session:
            session.delete(row(session, PersonaTable, persona))
            session.commit()

        with Session(db._engine) as session:
            assert row(session, BranchTable, branch).persona_id is None

    def test_database_cascades_branch_delete_to_messages(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)
        fill(db, branch, "a", "b")

        with Session(db._engine) as session:
            session.execute(delete(BranchTable).where(BranchTable.id == branch))
            session.commit()
            assert session.query(MessageTable).count() == 0

    def test_every_connection_enforces_foreign_keys(self, db: SqlAlchemy) -> None:
        for _ in range(3):
            with db._engine.connect() as conn:
                assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
