from typing import TYPE_CHECKING

from sensai.llm.responses import ChatResponse

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sensai.llm.provider_manager import ProviderManager
    from sensai.tools.tool_manager import ToolManager

THINK_INSTRUCTION = (
    "Do not answer yet. Think step by step, in two or three short sentences: what is being asked "
    "and what you already know for sure."
)
TOOLS_INSTRUCTION = (
    "Then decide whether you need one of the tools below. Your knowledge is old, so facts that may have changed "
    "(people, software versions, news, prices) always need a tool, even if you think you know them. "
    "Greetings, arithmetic and general knowledge need no tool. End with 'Tool needed: yes' or 'Tool needed: no'."
)
OBSERVE_INSTRUCTION = (
    "Do not answer yet. Read the tool results above and say, in one or two short sentences, what they tell you. "
    "End with 'Answer found: yes' if they are enough to answer the user, or 'Answer found: no' if something is "
    "still missing."
)
ACT_INSTRUCTION = (
    "Now act on your reasoning: call a tool if you said you need one. Otherwise reply to the user's last message "
    "directly, in the user's language, without repeating your reasoning."
)
ANSWER_INSTRUCTION = "Now reply to the user's last message directly, in the user's language, from what you found."
TOOL_REFUSAL = "tool needed: no"
SEARCH_AGAIN = "answer found: no"


class ReAct:
    """Answer with the ReAct method: the model thinks, acts with a tool, observes the result, then starts again."""

    def __init__(self, provider_manager: ProviderManager, tool_manager: ToolManager, max_iterations: int = 5) -> None:
        """Bind the ReAct loop to a `provider_manager` and a `tool_manager`, with a maximum of `max_iterations`."""
        self._provider_manager = provider_manager
        self._tool_manager = tool_manager
        self._max_iterations = max_iterations

    @property
    def max_iterations(self) -> int:
        """Maximum number of iterations the ReAct loop does before answering."""
        return self._max_iterations

    def update_max_iterations(self, max_iterations: int) -> None:
        """Change the maximum number of iterations the ReAct loop will do before giving up."""
        self._max_iterations = max_iterations

    def generate_answer(self, messages: list[dict]) -> Iterator[ChatResponse]:
        """Stream the answer to `messages`, repeating Think, Act and Observe until the model answers.

        Each step appends what it produced to `messages`, so the next step can read it.
        """
        if self._max_iterations == 0:
            yield from self._request(messages, ANSWER_INSTRUCTION, tools=[])
            return

        tools = self._available_tools()
        instruction = self._first_think_instruction(tools)
        observed = False

        for _ in range(self._max_iterations):
            yield from self._think(messages, instruction)
            thought = messages[-1]["content"]

            offered_tools = tools if self._wants_tool(thought, after_observation=observed) else []
            yield from self._act(messages, offered_tools)

            tool_calls = messages[-1].get("tool_calls", [])
            if not tool_calls:
                return

            yield from self._observe(messages, tool_calls)
            instruction = OBSERVE_INSTRUCTION
            observed = True

        yield from self._request(messages, ANSWER_INSTRUCTION, tools=[])

    def _think(self, messages: list[dict], instruction: str) -> Iterator[ChatResponse]:
        """Make the model reason without answering, and show its reasoning as thinking."""
        thought = ""

        for chunk in self._request(messages, instruction, tools=[]):
            if chunk.error is not None:
                yield chunk
            if chunk.content is not None:
                thought += chunk.content
                yield self._build_thinking(chunk.content)
        yield self._build_thinking("\n")
        messages.append({"role": "assistant", "content": thought})

    def _act(self, messages: list[dict], tools: list[dict]) -> Iterator[ChatResponse]:
        """Let the model call one of `tools`, or make it write the final answer when there are none."""
        instruction = ACT_INSTRUCTION if tools else ANSWER_INSTRUCTION
        answer = ""
        tool_calls = []

        for chunk in self._request(messages, instruction, tools):
            yield chunk
            if chunk.content is not None:
                answer += chunk.content
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)

        message: dict[str, object] = {"role": "assistant", "content": answer}
        if tool_calls:
            message["tool_calls"] = tool_calls
        messages.append(message)

    def _observe(self, messages: list[dict], tool_calls: list[dict]) -> Iterator[ChatResponse]:
        """Run each tool the model called and add its result to `messages`, showing the tool name as thinking."""
        for call in tool_calls:
            name = call.get("function", {}).get("name", "unknown")
            yield self._build_thinking(f"-> Using {name}...\n")

            result = self._tool_manager.call(call)
            messages.append(result)

    def _request(self, messages: list[dict], instruction: str, tools: list[dict]) -> Iterator[ChatResponse]:
        """Send `messages` followed by `instruction` to the model, offering it `tools` when there are some."""
        provider = self._provider_manager.provider()
        request_messages = messages.copy()
        request_messages.append({"role": "user", "content": instruction})
        # Only enables our thinking loop
        payload = {"model": provider.current_model(), "messages": request_messages, "think": False}

        if tools:
            payload["tools"] = tools
        return provider.chat(payload, stream=True)

    def _available_tools(self) -> list[dict]:
        """Return the tools to offer to the current model, none when it cannot call tools."""
        provider = self._provider_manager.provider()
        capabilities = provider.show(provider.current_model())

        if not capabilities.tools:
            return []
        return self._tool_manager.get_tool_call_rules()

    @staticmethod
    def _wants_tool(thought: str, *, after_observation: bool) -> bool:
        """Read the decision ending `thought`: a tool is the default before any tool result, the answer after one."""
        thought = thought.lower()

        if after_observation:
            return SEARCH_AGAIN in thought
        return TOOL_REFUSAL not in thought

    @staticmethod
    def _first_think_instruction(tools: list[dict]) -> str:
        """Build the instruction of the first Think step, listing the tools the model can plan to use."""
        if not tools:
            return THINK_INSTRUCTION

        lines = [THINK_INSTRUCTION, TOOLS_INSTRUCTION]
        for tool in tools:
            function = tool["function"]
            lines.append(f"- {function['name']}: {function['description']}")
        return "\n".join(lines)

    @staticmethod
    def _build_thinking(text: str) -> ChatResponse:
        """Build a chunk that the UI shows as thinking instead of as the answer."""
        return ChatResponse({"done": False, "message": {"thinking": text}})
