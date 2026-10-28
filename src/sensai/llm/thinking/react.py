from contextlib import closing
from typing import TYPE_CHECKING

from sensai.error import SensaiError
from sensai.llm.responses import ChatResponse
from sensai.monitoring.interaction import InteractionStatus, ignore_interaction
from sensai.monitoring.recorder import InteractionRecorder

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator

    from sensai.llm.provider_manager import ProviderManager
    from sensai.monitoring.interaction import Interaction
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

    def __init__(
        self,
        provider_manager: ProviderManager,
        tool_manager: ToolManager,
        max_iterations: int = 5,
        *,
        context_window: int | None = None,
    ) -> None:
        """Bind the ReAct loop to a `provider_manager` and a `tool_manager`, with a maximum of `max_iterations`.

        `context_window` is the number of tokens the model is asked to attend to; None leaves it to the provider.
        """
        self._provider_manager = provider_manager
        self._tool_manager = tool_manager
        self._max_iterations = max_iterations
        self._context_window = context_window

    @property
    def max_iterations(self) -> int:
        """Maximum number of iterations the ReAct loop does before answering."""
        return self._max_iterations

    def update_max_iterations(self, max_iterations: int) -> None:
        """Change the maximum number of iterations the ReAct loop will do before giving up."""
        self._max_iterations = max_iterations

    def generate_answer(
        self, messages: list[dict], on_interaction: Callable[[Interaction], None] = ignore_interaction
    ) -> Generator[ChatResponse]:
        """Stream the answer to `messages`, repeating Think, Act and Observe until the model answers.

        Each step appends what it produced to `messages`, so the next step can read it.
        Every call to the model is passed to `on_interaction` once over, whatever its outcome.
        """
        if self._max_iterations == 0:
            yield from self._request(messages, ANSWER_INSTRUCTION, tools=[], on_interaction=on_interaction)
            return

        tools = self._available_tools()
        instruction = self._first_think_instruction(tools)
        observed = False

        for _ in range(self._max_iterations):
            yield from self._think(messages, instruction, on_interaction)
            thought = messages[-1]["content"]

            offered_tools = tools if self._wants_tool(thought, after_observation=observed) else []
            yield from self._act(messages, offered_tools, on_interaction)

            tool_calls = messages[-1].get("tool_calls", [])
            if not tool_calls:
                return

            yield from self._observe(messages, tool_calls)
            instruction = OBSERVE_INSTRUCTION
            observed = True

        yield from self._request(messages, ANSWER_INSTRUCTION, tools=[], on_interaction=on_interaction)

    def _think(
        self, messages: list[dict], instruction: str, on_interaction: Callable[[Interaction], None]
    ) -> Iterator[ChatResponse]:
        """Make the model reason without answering, and show its reasoning as thinking."""
        thought = ""

        with closing(self._request(messages, instruction, tools=[], on_interaction=on_interaction)) as chunks:
            for chunk in chunks:
                if chunk.error is not None:
                    yield chunk
                if chunk.content is not None:
                    thought += chunk.content
                    yield self._build_thinking(chunk.content)
        yield self._build_thinking("\n")
        messages.append({"role": "assistant", "content": thought})

    def _act(
        self, messages: list[dict], tools: list[dict], on_interaction: Callable[[Interaction], None]
    ) -> Iterator[ChatResponse]:
        """Let the model call one of `tools`, or make it write the final answer when there are none."""
        instruction = ACT_INSTRUCTION if tools else ANSWER_INSTRUCTION
        answer = ""
        tool_calls = []

        with closing(self._request(messages, instruction, tools, on_interaction)) as chunks:
            for chunk in chunks:
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
            name = self._tool_manager.tool_name(call)
            yield self._build_thinking(f"-> Using {name}...\n")

            result = self._tool_manager.call(call)
            messages.append(result)

    def _request(
        self, messages: list[dict], instruction: str, tools: list[dict], on_interaction: Callable[[Interaction], None]
    ) -> Generator[ChatResponse]:
        """Send `messages` followed by `instruction` to the model, offering it `tools` when there are some.

        The call is recorded and passed to `on_interaction` when it ends: done, failed, or closed before its end.
        """
        provider = self._provider_manager.provider()
        model = provider.current_model()
        request_messages = messages.copy()
        request_messages.append({"role": "user", "content": instruction})
        # Only enables our thinking loop
        payload = {"model": model, "messages": request_messages, "think": False}

        if tools:
            payload["tools"] = tools
        if self._context_window is not None:
            payload["options"] = {"num_ctx": self._context_window}
        recorder = InteractionRecorder(model=model, request=request_messages)
        status, error = InteractionStatus.OK, None
        try:
            for chunk in provider.chat(payload, stream=True):
                recorder.add_chunk(chunk)
                if chunk.error is not None:
                    status, error = InteractionStatus.ERROR, chunk.error
                yield chunk
        except GeneratorExit:
            if status is InteractionStatus.OK:
                status = InteractionStatus.INTERRUPTED
            raise
        except SensaiError as err:
            status, error = InteractionStatus.ERROR, str(err)
            raise
        finally:
            on_interaction(recorder.finish(status, error))

    def _available_tools(self) -> list[dict]:
        """Return the tools to offer to the current model, none when there are none or when it cannot call tools.

        The capabilities of the model are only looked up when there are tools to offer.
        """
        rules = self._tool_manager.get_tool_call_rules()
        if not rules:
            return []

        provider = self._provider_manager.provider()
        capabilities = provider.show(provider.current_model())
        if not capabilities.tools:
            return []
        return rules

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
