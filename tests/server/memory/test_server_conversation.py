from pathlib import Path

import pytest
from sensai_protocol import System
from sensai_protocol.error import SensaiError
from sensai_protocol.server_messages import Role
from sensai_server.llm.persona import Persona
from sensai_server.memory import Conversation
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy

from tests.server.conftest import MODEL, db_url
from tests.server.fakes import RecordingOutbound


def test_a_new_conversation_has_no_persona() -> None:
    assert Conversation().persona is None


def test_each_new_conversation_has_its_own_rag_id() -> None:
    first, second = Conversation().rag_id, Conversation().rag_id
    assert first is not None
    assert second is not None
    assert first != second


@pytest.fixture
def db(tmp_path: Path) -> SqlAlchemy:
    return SqlAlchemy(db_url(tmp_path))


def _started(db: SqlAlchemy) -> Conversation:
    conv = Conversation()
    conv.model = MODEL
    conv.add_conv_to_db(db)
    return conv


class TestReset:
    def test_leaves_the_branch_with_an_empty_history_and_a_new_rag_id(self, db: SqlAlchemy) -> None:
        conv = _started(db)
        conv.history.append(Role.USER, "hi")
        rag_id = conv.rag_id

        conv.reset()

        assert conv.uuid is None
        assert conv.history.messages() == []
        assert conv.rag_id not in {None, rag_id}

    def test_keeps_the_model_and_the_persona(self) -> None:
        conv = Conversation()
        conv.model = MODEL
        conv.persona = Persona(None, "analyst", "Analyses things", "You are an analyst.")

        conv.reset()

        assert conv.model == MODEL
        assert conv.persona is not None

    def test_the_branch_left_stays_in_the_database(self, db: SqlAlchemy) -> None:
        conv = _started(db)
        branch = conv.uuid

        conv.reset()

        assert branch in db.get_branches()


class TestAddConvToDb:
    def test_creates_a_branch_on_the_model_linked_to_the_rag_id(self, db: SqlAlchemy) -> None:
        conv = _started(db)

        assert conv.uuid is not None
        assert list(db.get_branches()) == [conv.uuid]
        assert db.get_branch_rag_id(conv.uuid) == conv.rag_id

    def test_takes_the_given_name(self, db: SqlAlchemy) -> None:
        conv = Conversation()
        conv.model = MODEL

        conv.add_conv_to_db(db, name="Trip")

        assert db.get_branches() == {conv.uuid: "Trip"}

    def test_without_a_model_raises_and_creates_nothing(self, db: SqlAlchemy) -> None:
        conv = Conversation()

        with pytest.raises(SensaiError):
            conv.add_conv_to_db(db)

        assert db.get_branches() == {}
        assert conv.uuid is None


class TestSaveMessage:
    async def test_stores_the_message_and_keeps_it_in_the_history(self, db: SqlAlchemy) -> None:
        conv = _started(db)
        out = RecordingOutbound()

        assert await conv.save_message(Role.USER, "hi", out, db)

        assert conv.uuid is not None
        [stored] = db.get_messages(conv.uuid)
        assert conv.history.messages() == [stored]
        assert out.sent == []

    async def test_a_conversation_not_started_stores_nothing(self, db: SqlAlchemy) -> None:
        conv = Conversation()
        out = RecordingOutbound()

        assert not await conv.save_message(Role.USER, "hi", out, db)

        assert conv.history.messages() == []
        assert out.sent == [System(message="Failed to add message")]

    async def test_a_removed_branch_is_reported_and_stores_nothing(self, db: SqlAlchemy) -> None:
        conv = _started(db)
        assert conv.uuid is not None
        db.remove_branch(conv.uuid)
        out = RecordingOutbound()

        assert not await conv.save_message(Role.USER, "hi", out, db)

        assert conv.history.messages() == []
        [message] = out.sent
        assert isinstance(message, System)
        assert "InvalidInstanceError" in message.message

    async def test_a_failed_insert_is_reported(self, db: SqlAlchemy, monkeypatch: pytest.MonkeyPatch) -> None:
        conv = _started(db)
        monkeypatch.setattr(db, "add_message_to_branch", lambda *_: None)
        out = RecordingOutbound()

        assert not await conv.save_message(Role.USER, "hi", out, db)

        assert conv.history.messages() == []
        assert out.sent == [System(message="Failed to add message")]
