from sensai.llm.message import Message, Role
from sensai.memory.history import History
from sensai.memory.history_repository import HistoryRepository


def make_history() -> HistoryRepository:
    return History()


class TestHistory:
    def test_starts_empty(self) -> None:
        assert make_history().messages() == []

    def test_keeps_messages_in_append_order(self) -> None:
        history = make_history()
        history.append(Role.SYSTEM, "be concise")
        history.append(Role.USER, "hi")
        history.append(Role.ASSISTANT, "hello")

        assert history.messages() == [
            Message(Role.SYSTEM, "be concise"),
            Message(Role.USER, "hi"),
            Message(Role.ASSISTANT, "hello"),
        ]

    def test_messages_returns_a_copy(self) -> None:
        history = make_history()
        history.append(Role.USER, "hi")

        history.messages().clear()

        assert history.messages() == [Message(Role.USER, "hi")]

    def test_clear_removes_every_message(self) -> None:
        history = make_history()
        history.append(Role.USER, "hi")

        history.clear()

        assert history.messages() == []

    def test_to_json_maps_onto_chat_payload(self) -> None:
        history = make_history()
        history.append(Role.USER, "hi")
        history.append(Role.ASSISTANT, "hello")

        assert history.to_json() == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_to_json_of_empty_history_is_empty(self) -> None:
        assert make_history().to_json() == []
