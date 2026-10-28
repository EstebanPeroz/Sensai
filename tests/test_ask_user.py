import pytest

from sensai.tools.ask_user import NO_ANSWER, AskUserTool
from sensai.tools.tool_manager import ToolManager
from tests.commands.fakes import FakeUI


def make_call(arguments: dict) -> dict:
    return {"type": "function", "function": {"name": "ask_user", "arguments": arguments}}


class TestAskUser:
    def test_returns_the_answer_of_the_user(self) -> None:
        ui = FakeUI()
        ui.answers = ["Q2 2026"]

        assert AskUserTool(ui.ask_user).run(question="Which quarter?") == "Q2 2026"
        assert ui.questions == ["Which quarter?"]

    def test_answer_is_stripped(self) -> None:
        assert AskUserTool(lambda _question: "  Q2  ").run(question="Which quarter?") == "Q2"

    @pytest.mark.parametrize("answer", [None, "", "   "])
    def test_missing_answer_asks_the_model_to_state_its_assumption(self, answer: str | None) -> None:
        assert AskUserTool(lambda _question: answer).run(question="Which quarter?") == NO_ANSWER

    def test_missing_question_is_an_error_for_the_model(self) -> None:
        ui = FakeUI()
        result = ToolManager([AskUserTool(ui.ask_user)]).call(make_call({}))

        assert result["content"].startswith("error:")
        assert ui.questions == []

    def test_does_not_require_confirmation(self) -> None:
        assert AskUserTool(FakeUI().ask_user).requires_confirmation is False

    def test_question_is_a_required_string(self) -> None:
        parameters = AskUserTool(FakeUI().ask_user).parameters

        assert parameters["required"] == ["question"]
        assert parameters["properties"]["question"]["type"] == "string"
