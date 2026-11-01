import pytest
from sensai_server.tools.ask_user import NO_ANSWER, AskUserTool
from sensai_server.tools.tool import ToolContext
from sensai_server.tools.tool_manager import ToolManager

from tests.server.fakes import RecordingOutbound


class AnsweringOutbound(RecordingOutbound):
    """Answers every question of the model with `answer`, recording the questions asked."""

    def __init__(self, answer: str | None = None) -> None:
        super().__init__()
        self._answer = answer
        self.questions: list[str] = []

    async def ask_user(self, question: str) -> str | None:
        self.questions.append(question)
        return self._answer


def make_call(arguments: dict) -> dict:
    return {"type": "function", "function": {"name": "ask_user", "arguments": arguments}}


async def ask(answer: str | None) -> str:
    return await AskUserTool().run(ToolContext(out=AnsweringOutbound(answer)), question="Which quarter?")


class TestAskUser:
    async def test_returns_the_answer_of_the_user(self) -> None:
        out = AnsweringOutbound("Q2 2026")

        assert await AskUserTool().run(ToolContext(out=out), question="Which quarter?") == "Q2 2026"
        assert out.questions == ["Which quarter?"]

    async def test_answer_is_stripped(self) -> None:
        assert await ask("  Q2  ") == "Q2"

    @pytest.mark.parametrize("answer", [None, "", "   "])
    async def test_missing_answer_asks_the_model_to_state_its_assumption(self, answer: str | None) -> None:
        assert await ask(answer) == NO_ANSWER

    async def test_missing_question_is_an_error_for_the_model(self) -> None:
        out = AnsweringOutbound("Q2")
        manager = ToolManager([AskUserTool()])

        result = await manager.call(make_call({}), manager.new_state(), ToolContext(out=out))

        assert result["content"].startswith("error:")
        assert out.questions == []

    async def test_asks_the_client_of_its_context_only(self) -> None:
        first, second = AnsweringOutbound("first"), AnsweringOutbound("second")
        tool = AskUserTool()

        assert await tool.run(ToolContext(out=first), question="Who?") == "first"
        assert await tool.run(ToolContext(out=second), question="Who?") == "second"
        assert (first.questions, second.questions) == (["Who?"], ["Who?"])

    def test_does_not_require_confirmation(self) -> None:
        assert AskUserTool().requires_confirmation is False

    def test_question_is_a_required_string(self) -> None:
        parameters = AskUserTool().parameters

        assert parameters["required"] == ["question"]
        assert parameters["properties"]["question"]["type"] == "string"
