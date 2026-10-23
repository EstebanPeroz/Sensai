from unittest.mock import MagicMock

from sensai.history.conversation import Conversation
from sensai.llm.message import Role
from sensai.llm.responses import ChatResponse
from sensai.llm.thinking.react import ReAct


def test_conversation_delegates_generation_to_react() -> None:
    provider_manager = MagicMock()
    persona_manager = MagicMock()
    persona_manager.active_persona.return_value = None
    react = MagicMock(spec=ReAct)
    react.generate_answer.return_value = iter(
        [ChatResponse({"message": {"role": "assistant", "content": "Hello"}})],
    )
    conversation = Conversation(provider_manager, persona_manager, react)

    def record_message(role: Role, content: str, ui: object) -> bool:
        del ui
        conversation.history.append(role, content)
        return True

    conversation.add_message = MagicMock(side_effect=record_message)
    ui = MagicMock()
    ui.wait_event.return_value = False

    conversation.chat("Bonjour", ui)

    react.generate_answer.assert_called_once_with([{"role": "user", "content": "Bonjour"}])
    assert [call.args[0].content for call in ui.send_ai_response.call_args_list] == ["Hello"]
    conversation.add_message.assert_called_with(Role.ASSISTANT, "Hello", ui)
