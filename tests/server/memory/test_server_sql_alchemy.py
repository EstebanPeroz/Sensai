from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sensai_protocol.server_messages import Role
from sensai_server.llm.provider import ProviderManager
from sensai_server.memory import Conversation, Message
from sensai_server.memory.persistent_db.error import DatabaseError, DbConnectionError, InvalidInstanceError
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai_server.memory.persistent_db.sql_alchemy.table import BranchTable, MessageTable, PersonaTable
from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from tests.server.conftest import MODEL, SETTINGS, db_url


@pytest.fixture
def db(tmp_path: Path) -> SqlAlchemy:
    return SqlAlchemy(db_url(tmp_path))


def new_branch(db: SqlAlchemy, source: UUID | None = None, model: str = MODEL) -> UUID:
    branch = db.create_branch(model=model, current_branch=source)
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


def make_persona(db: SqlAlchemy, name: str = "p") -> UUID:
    with Session(db._engine) as session:  # noqa: SLF001
        persona = PersonaTable(name=name, description="d", prompt="x")
        session.add(persona)
        session.commit()
        return persona.id


class TestInit:
    def test_creates_the_schema(self, db: SqlAlchemy) -> None:
        with Session(db._engine) as session:  # noqa: SLF001
            assert session.query(BranchTable).count() == 0
            assert session.query(MessageTable).count() == 0

    def test_creates_the_db_file(self, db: SqlAlchemy, tmp_path: Path) -> None:  # noqa: ARG002
        assert (tmp_path / "test.db").exists()

    def test_data_survives_a_new_instance(self, db: SqlAlchemy, tmp_path: Path) -> None:
        branch = new_branch(db)
        assert SqlAlchemy(db_url(tmp_path)).get_branches() == {branch: "Branch"}

    def test_invalid_url_raises(self) -> None:
        with pytest.raises(DbConnectionError):
            SqlAlchemy("not a url")


class TestCreateBranch:
    def test_new_empty_branch(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)

        assert db.get_messages(branch) == []
        assert db.get_branches() == {branch: "Branch"}

    def test_new_branch_takes_the_given_name(self, db: SqlAlchemy) -> None:
        branch = db.create_branch(model=MODEL, name="Imported")

        assert db.get_branches() == {branch: "Imported"}

    def test_new_branches_get_unique_names(self, db: SqlAlchemy) -> None:
        ids = [new_branch(db) for _ in range(3)]

        assert len(set(ids)) == 3
        assert sorted(db.get_branches().values()) == ["Branch", "Branch 2", "Branch 3"]

    def test_new_branch_keeps_the_model(self, db: SqlAlchemy) -> None:
        branch = new_branch(db, model="b")

        with Session(db._engine) as session:  # noqa: SLF001
            assert row(session, BranchTable, branch).model_name == "b"

    def test_copy_has_new_id_and_same_messages(self, db: SqlAlchemy) -> None:
        source = new_branch(db)
        fill(db, source, "a", "b", "c")

        copy = new_branch(db, source)

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

        assert sorted(db.get_branches().values()) == ["Branch", "Branch 2", "Branch 3"]

    def test_copy_of_a_copy_gets_a_free_name(self, db: SqlAlchemy) -> None:
        source = new_branch(db)
        second = new_branch(db, source)

        new_branch(db, second)

        assert sorted(db.get_branches().values()) == ["Branch", "Branch 2", "Branch 2 2"]

    def test_copy_keeps_model_name(self, db: SqlAlchemy) -> None:
        source = new_branch(db, model="b")

        copy = new_branch(db, source, model=MODEL)

        with Session(db._engine) as session:  # noqa: SLF001
            assert row(session, BranchTable, copy).model_name == "b"

    def test_copy_message_ids_are_new_and_linked_to_the_copy(self, db: SqlAlchemy) -> None:
        source = new_branch(db)
        fill(db, source, "a", "b")

        copy = new_branch(db, source)

        with Session(db._engine) as session:  # noqa: SLF001
            src_msgs = row(session, BranchTable, source).messages
            copy_msgs = row(session, BranchTable, copy).messages
            assert len(copy_msgs) == 2
            assert {m.id for m in copy_msgs}.isdisjoint({m.id for m in src_msgs})
            assert all(m.branch_id == copy for m in copy_msgs)
            assert session.query(MessageTable).count() == 4

    def test_unknown_source_returns_none(self, db: SqlAlchemy) -> None:
        assert db.create_branch(model=MODEL, current_branch=uuid4()) is None
        assert db.get_branches() == {}


class TestGetBranches:
    def test_empty(self, db: SqlAlchemy) -> None:
        assert db.get_branches() == {}

    def test_maps_id_to_name(self, db: SqlAlchemy) -> None:
        first = new_branch(db)
        second = new_branch(db)

        assert db.get_branches() == {first: "Branch", second: "Branch 2"}


class TestGetBranch:
    def test_loads_uuid_model_and_messages(self, db: SqlAlchemy) -> None:
        branch = new_branch(db, model="b")
        fill(db, branch, "hi", "hello")
        conv = Conversation()

        db.get_branch(conv, branch, ProviderManager(SETTINGS.llm))

        assert conv.uuid == branch
        assert conv.model == "b"
        assert turns(conv.history.messages()) == [(Role.USER, "hi"), (Role.ASSISTANT, "hello")]

    def test_loading_the_current_branch_changes_nothing(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)
        fill(db, branch, "hi")
        conv = Conversation()
        conv.uuid = branch

        db.get_branch(conv, branch, ProviderManager(SETTINGS.llm))

        assert conv.history.messages() == []

    def test_unknown_branch_raises_and_leaves_conversation_untouched(self, db: SqlAlchemy) -> None:
        conv = Conversation()
        conv.history.append(Role.USER, "kept")

        with pytest.raises(InvalidInstanceError):
            db.get_branch(conv, uuid4(), ProviderManager(SETTINGS.llm))

        assert conv.uuid is None
        assert conv.model is None
        assert turns(conv.history.messages()) == [(Role.USER, "kept")]

    def test_a_model_no_provider_serves_raises_and_leaves_conversation_untouched(self, db: SqlAlchemy) -> None:
        branch = new_branch(db, model="unserved")
        conv = Conversation()

        with pytest.raises(DatabaseError):
            db.get_branch(conv, branch, ProviderManager(SETTINGS.llm))

        assert conv.uuid is None
        assert conv.model is None


class TestSetBranchName:
    def test_renames_the_branch(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)

        db.set_branch_name(branch, "renamed")

        assert db.get_branches() == {branch: "renamed"}

    def test_only_renames_that_branch(self, db: SqlAlchemy) -> None:
        keep = new_branch(db)
        renamed = new_branch(db)

        db.set_branch_name(renamed, "renamed")

        assert db.get_branches() == {keep: "Branch", renamed: "renamed"}

    def test_unknown_branch_raises(self, db: SqlAlchemy) -> None:
        with pytest.raises(InvalidInstanceError):
            db.set_branch_name(uuid4(), "renamed")


class TestSetBranchModel:
    def test_sets_the_model(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)

        db.set_branch_model(branch, "llama")

        with Session(db._engine) as session:  # noqa: SLF001
            assert row(session, BranchTable, branch).model_name == "llama"

    def test_unknown_branch_raises(self, db: SqlAlchemy) -> None:
        with pytest.raises(InvalidInstanceError):
            db.set_branch_model(uuid4(), "llama")


class TestRemoveBranch:
    def test_removes_only_that_branch(self, db: SqlAlchemy) -> None:
        keep = new_branch(db)
        drop = new_branch(db)

        db.remove_branch(drop)

        assert db.get_branches() == {keep: "Branch"}

    def test_cascades_to_messages(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)
        fill(db, branch, "a", "b")

        db.remove_branch(branch)

        with Session(db._engine) as session:  # noqa: SLF001
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

    def test_returns_the_id_of_the_stored_message(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)

        message_id = db.add_message_to_branch(branch, Role.USER, "hi")

        assert [m.uuid for m in db.get_messages(branch)] == [message_id]

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

        with Session(db._engine) as session:  # noqa: SLF001
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


class TestSetBranchPersona:
    def test_sets_the_persona(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)
        persona = make_persona(db)

        assert db.set_branch_persona(branch, persona) is True

        with Session(db._engine) as session:  # noqa: SLF001
            assert row(session, BranchTable, branch).persona_id == persona
            assert [b.id for b in row(session, PersonaTable, persona).branches] == [branch]

    def test_can_switch_persona(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)
        first, second = make_persona(db, "a"), make_persona(db, "b")
        db.set_branch_persona(branch, first)

        assert db.set_branch_persona(branch, second) is True

        with Session(db._engine) as session:  # noqa: SLF001
            assert row(session, BranchTable, branch).persona_id == second
            assert row(session, PersonaTable, first).branches == []

    def test_unknown_persona_fails_and_changes_nothing(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)
        persona = make_persona(db)
        db.set_branch_persona(branch, persona)

        assert db.set_branch_persona(branch, uuid4()) is False

        with Session(db._engine) as session:  # noqa: SLF001
            assert row(session, BranchTable, branch).persona_id == persona

    def test_unknown_branch_fails(self, db: SqlAlchemy) -> None:
        assert db.set_branch_persona(uuid4(), make_persona(db)) is False


class TestForeignKeys:
    def test_message_with_unknown_branch_is_rejected(self, db: SqlAlchemy) -> None:
        with Session(db._engine) as session:  # noqa: SLF001
            session.add(MessageTable(branch_id=uuid4(), role=Role.USER, content="x"))
            with pytest.raises(IntegrityError):
                session.commit()

    def test_branch_with_unknown_persona_is_rejected(self, db: SqlAlchemy) -> None:
        with Session(db._engine) as session:  # noqa: SLF001
            session.add(BranchTable(name="b", model_name="", persona_id=uuid4()))
            with pytest.raises(IntegrityError):
                session.commit()

    def test_deleting_a_persona_unsets_it_on_branches(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)
        persona = make_persona(db)
        db.set_branch_persona(branch, persona)

        with Session(db._engine) as session:  # noqa: SLF001
            session.delete(row(session, PersonaTable, persona))
            session.commit()

        with Session(db._engine) as session:  # noqa: SLF001
            assert row(session, BranchTable, branch).persona_id is None

    def test_database_cascades_branch_delete_to_messages(self, db: SqlAlchemy) -> None:
        branch = new_branch(db)
        fill(db, branch, "a", "b")

        with Session(db._engine) as session:  # noqa: SLF001
            session.execute(delete(BranchTable).where(BranchTable.id == branch))
            session.commit()
            assert session.query(MessageTable).count() == 0

    def test_every_connection_enforces_foreign_keys(self, db: SqlAlchemy) -> None:
        for _ in range(3):
            with db._engine.connect() as conn:  # noqa: SLF001
                assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
