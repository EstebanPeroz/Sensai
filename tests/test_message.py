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
