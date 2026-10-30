from sensai_server.memory import Conversation


def test_a_new_conversation_has_no_persona() -> None:
    assert Conversation().persona is None
