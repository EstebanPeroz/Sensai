from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest
from sensai_protocol.server_messages import Role
from sensai_server.memory import History, Message

TOOL_CALLS = [{"type": "function", "function": {"name": "echo", "arguments": {"city": "Paris"}}}]


class TestMessage:
    @pytest.mark.parametrize("role", list(Role))
    def test_to_dict_maps_onto_chat_message(self, role: Role) -> None:
        assert Message(None, role, "hello").to_dict() == {"role": role.value, "content": "hello"}

    def test_roles_match_chat_endpoint(self) -> None:
        assert [role.value for role in Role] == ["system", "user", "assistant", "tool"]

    def test_is_immutable(self) -> None:
        message = Message(None, Role.USER, "hi")

        with pytest.raises(FrozenInstanceError):
            message.content = "changed"  # type: ignore[misc]

    def test_to_dict_leaves_the_uuid_out(self) -> None:
        assert Message(uuid4(), Role.USER, "hi").to_dict() == {"role": "user", "content": "hi"}

    def test_to_dict_includes_tool_calls_of_assistant_message(self) -> None:
        message = Message(None, Role.ASSISTANT, "", tool_calls=TOOL_CALLS)

        assert message.to_dict() == {"role": "assistant", "content": "", "tool_calls": TOOL_CALLS}

    def test_to_dict_includes_tool_name_of_tool_message(self) -> None:
        message = Message(None, Role.TOOL, "Paris", tool_name="echo")

        assert message.to_dict() == {"role": "tool", "content": "Paris", "tool_name": "echo"}

    def test_to_dict_omits_empty_tool_calls(self) -> None:
        message = Message(None, Role.ASSISTANT, "hello", tool_calls=[])

        assert message.to_dict() == {"role": "assistant", "content": "hello"}


class TestHistory:
    def test_starts_empty(self) -> None:
        assert History().messages() == []

    def test_keeps_messages_in_append_order(self) -> None:
        history = History()
        history.append(Role.SYSTEM, "be concise")
        history.append(Role.USER, "hi")
        history.append(Role.ASSISTANT, "hello")

        assert history.messages() == [
            Message(None, Role.SYSTEM, "be concise"),
            Message(None, Role.USER, "hi"),
            Message(None, Role.ASSISTANT, "hello"),
        ]

    def test_insert_keeps_the_message_as_given(self) -> None:
        history = History()
        message = Message(uuid4(), Role.USER, "hi")

        history.insert(message)

        assert history.messages() == [message]

    def test_messages_returns_a_copy(self) -> None:
        history = History()
        history.append(Role.USER, "hi")

        history.messages().clear()

        assert history.messages() == [Message(None, Role.USER, "hi")]

    def test_clear_removes_every_message(self) -> None:
        history = History()
        history.append(Role.USER, "hi")

        history.clear()

        assert history.messages() == []

    def test_replace_swaps_every_message(self) -> None:
        history = History()
        history.append(Role.USER, "old")
        new = [Message(uuid4(), Role.USER, "hi"), Message(uuid4(), Role.ASSISTANT, "hello")]

        history.replace(new)

        assert history.messages() == new

    def test_to_json_maps_onto_chat_payload(self) -> None:
        history = History()
        history.append(Role.USER, "hi")
        history.append(Role.ASSISTANT, "hello")

        assert history.to_json() == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_to_json_of_empty_history_is_empty(self) -> None:
        assert History().to_json() == []

    def test_keeps_tool_calls_and_tool_name(self) -> None:
        history = History()
        history.append(Role.ASSISTANT, "", tool_calls=TOOL_CALLS)
        history.append(Role.TOOL, "Paris", tool_name="echo")

        assert history.messages() == [
            Message(None, Role.ASSISTANT, "", tool_calls=TOOL_CALLS),
            Message(None, Role.TOOL, "Paris", tool_name="echo"),
        ]

    def test_to_json_includes_tool_calls_and_tool_name(self) -> None:
        history = History()
        history.append(Role.USER, "where am I?")
        history.append(Role.ASSISTANT, "", tool_calls=TOOL_CALLS)
        history.append(Role.TOOL, "Paris", tool_name="echo")

        assert history.to_json() == [
            {"role": "user", "content": "where am I?"},
            {"role": "assistant", "content": "", "tool_calls": TOOL_CALLS},
            {"role": "tool", "content": "Paris", "tool_name": "echo"},
        ]
