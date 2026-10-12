from dataclasses import FrozenInstanceError

import pytest

from sensai.llm.message import Message, Role


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

    def test_to_dict_includes_tool_calls_of_assistant_message(self) -> None:
        tool_calls = [{"type": "function", "function": {"name": "echo", "arguments": {"city": "Paris"}}}]

        message = Message(None, Role.ASSISTANT, "", tool_calls=tool_calls)

        assert message.to_dict() == {"role": "assistant", "content": "", "tool_calls": tool_calls}

    def test_to_dict_includes_tool_name_of_tool_message(self) -> None:
        message = Message(None, Role.TOOL, "Paris", tool_name="echo")

        assert message.to_dict() == {"role": "tool", "content": "Paris", "tool_name": "echo"}

    def test_to_dict_omits_empty_tool_calls(self) -> None:
        message = Message(None, Role.ASSISTANT, "hello", tool_calls=[])

        assert message.to_dict() == {"role": "assistant", "content": "hello"}
