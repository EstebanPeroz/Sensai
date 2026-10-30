from __future__ import annotations

from typing import TYPE_CHECKING

from sensai_server.tools.tool import Tool, get_parameter

if TYPE_CHECKING:
    from sensai_server.tools.tool import ToolContext

NO_ANSWER = "The user did not answer. State the assumption you make in your answer."


class AskUserTool(Tool[str]):
    """Lets the model ask the user for information missing from the request, instead of guessing it."""

    @property
    def name(self) -> str:
        """Name the model calls this tool by."""
        return "ask_user"

    @property
    def description(self) -> str:
        """What the tool does, read by the model to decide when to call it."""
        return (
            "Ask the user one short question when their request is missing information that no other tool can "
            "find, such as which period, which file or which product they mean. Returns their answer. To use when"
            " you are unsure what the user wants, and you need to ask them for clarification."
        )

    @property
    def parameters(self) -> dict:
        """JSON schema of the arguments accepted by `run`."""
        return {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "The question to ask the user."}},
            "required": ["question"],
        }

    async def run(self, ctx: ToolContext, **kwargs: object) -> str:
        """Ask `question` to the client of `ctx` and return the answer, or tell the model to make an assumption."""
        question = get_parameter("question", str, **kwargs)
        answer = await ctx.out.ask_user(question)
        if answer is None or not answer.strip():
            return NO_ANSWER
        return answer.strip()
