from unittest.mock import MagicMock

from sensai.history.conversation import Conversation
from sensai.llm.message import Role
from sensai.llm.responses import ChatResponse
from sensai.tools.tool_manager import ToolManager


def test_model_receives_mcp_result_and_produces_final_text() -> None:
    provider = MagicMock()
    tool_call = {
        "type": "function",
        "function": {"name": "add", "arguments": {"left": 2, "right": 2}},
    }
    provider.chat.side_effect = [
        iter([ChatResponse({"message": {"role": "assistant", "tool_calls": [tool_call]}})]),
        iter([ChatResponse({"message": {"role": "assistant", "content": "The answer is 4"}})]),
    ]
    provider.current_model.return_value = "test"
    provider_manager = MagicMock()
    provider_manager.provider.return_value = provider
    tool_manager = MagicMock(spec=ToolManager)
    tool_manager.call.return_value = {"role": "tool", "tool_name": "add", "content": "4"}
    tool_manager.get_tool_call_rules.return_value = []
    persona_manager = MagicMock()
    persona_manager.active_persona.return_value = None
    conversation = Conversation(provider_manager, persona_manager, tool_manager)

    def record_message(role: Role, content: str, ui: object) -> bool:
        del ui
        conversation.history.append(role, content)
        return True

    conversation.add_message = MagicMock(
        side_effect=record_message,
    )
    ui = MagicMock()
    ui.wait_event.return_value = False

    conversation.chat("What is 2 + 2?", ui)

    assert provider.chat.call_count == 2
    assert provider.chat.call_args_list[0].args[0] == {
        "model": provider_manager.provider.return_value.current_model.return_value,
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    }
    assert provider.chat.call_args_list[0].kwargs == {"stream": True}
    assert provider.chat.call_args_list[1].args[0] == {
        "model": provider_manager.provider.return_value.current_model.return_value,
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "tool_calls": [tool_call]},
            {"role": "tool", "content": "4"},
        ],
    }
    assert provider.chat.call_args_list[1].kwargs == {"stream": True}
    tool_manager.call.assert_called_once_with(tool_call)
    assert [call.args[0].content for call in ui.send_ai_response.call_args_list] == [None, "The answer is 4"]
    conversation.add_message.assert_any_call(Role.USER, "What is 2 + 2?", ui)
    conversation.add_message.assert_called_with(Role.ASSISTANT, "The answer is 4", ui)
    assert conversation.history.to_json() == [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "The answer is 4"},
    ]
