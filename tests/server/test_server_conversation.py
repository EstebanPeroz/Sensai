from sensai_server.memory import Conversation


def test_a_new_conversation_has_no_persona() -> None:
    assert Conversation().persona is None


def test_each_new_conversation_has_its_own_rag_id() -> None:
    first, second = Conversation().rag_id, Conversation().rag_id
    assert first is not None
    assert second is not None
    assert first != second
