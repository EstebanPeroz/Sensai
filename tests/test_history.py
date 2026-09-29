from sensai.llm.message import Message, Role
from sensai.memory.history import InMemoryHistory
from sensai.memory.repository import HistoryRepository


def make_history() -> HistoryRepository:
    return InMemoryHistory()


class TestInMemoryHistory:
    def test_starts_empty(self) -> None:
        assert make_history().messages() == []

    def test_keeps_messages_in_append_order(self) -> None:
        history = make_history()
        turns = [
            Message(Role.SYSTEM, "be concise"),
            Message(Role.USER, "hi"),
            Message(Role.ASSISTANT, "hello"),
        ]
        for turn in turns:
            history.append(turn)

        assert history.messages() == turns

    def test_messages_returns_a_copy(self) -> None:
        history = make_history()
        history.append(Message(Role.USER, "hi"))

        history.messages().clear()

        assert history.messages() == [Message(Role.USER, "hi")]

    def test_clear_removes_every_message(self) -> None:
        history = make_history()
        history.append(Message(Role.USER, "hi"))

        history.clear()

        assert history.messages() == []

    def test_messages_map_onto_chat_payload(self) -> None:
        history = make_history()
        history.append(Message(Role.USER, "hi"))
        history.append(Message(Role.ASSISTANT, "hello"))

        assert [message.to_dict() for message in history.messages()] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
