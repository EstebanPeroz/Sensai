"""The ReAct loop: think, act and observe until the model answers."""

from __future__ import annotations

import asyncio
import re
from contextlib import aclosing
from functools import partial
from typing import TYPE_CHECKING

from sensai_protocol.error import SensaiError

from sensai_server.llm.responses import ChatResponse
from sensai_server.llm.threaded import iterate_in_thread
from sensai_server.monitoring.interaction import InteractionStatus
from sensai_server.monitoring.recorder import InteractionRecorder
from sensai_server.tools.tool import ToolContext

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sensai_server.llm.provider import ProviderAdapter, ProviderManager
    from sensai_server.pipeline.chain import ChatContext
    from sensai_server.tools.tool_manager import ToolManager

EFFORTS = {"off": 0, "low": 1, "medium": 3, "high": 5}
"""The thinking efforts a session can pick with `/effort`, and how many ReAct steps each allows."""

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
# The decision ending a thought, such as "Tool needed: yes" or "Answer found: **no**".
TOOL_DECISION = re.compile(r"tool needed\W*(yes|no)", re.IGNORECASE)
ANSWER_DECISION = re.compile(r"answer found\W*(yes|no)", re.IGNORECASE)


class ReAct:
    """Answer with the ReAct method: the model thinks, acts with a tool, observes the result, then starts again.

    It keeps nothing between two exchanges: the model, the messages, the tools, the outbound and the number of
    steps all come from the `ChatContext`, so one instance serves every session at once.
    """

    def __init__(self, provider_manager: ProviderManager, tool_manager: ToolManager) -> None:
        """Bind the ReAct loop to the `provider_manager` serving the models and the `tool_manager` it calls."""
        self._provider_manager = provider_manager
        self._tool_manager = tool_manager

    async def generate_answer(self, context: ChatContext) -> AsyncGenerator[ChatResponse]:
        """Stream the answer to `context`, repeating Think, Act and Observe until the model answers.

        Each step appends what it produced to `context.messages`, so the next step can read it. That list is the
        copy `run_chat` built from the history: the thoughts, tool calls and tool results are never saved, only
        the answer streamed to the client is. Only the chunk ending the answer is `done`. Raises `SensaiError`
        when no provider serves the model.
        """
        provider = self._provider_manager.get_provider(context.model)
        if provider is None:
            msg = f"No provider serves model {context.model}"
            raise SensaiError(msg)

        if context.max_iterations == 0:
            async with aclosing(self._request(context, provider, ANSWER_INSTRUCTION, [])) as chunks:
                async for chunk in chunks:
                    yield chunk
            return

        tools = await self._available_tools(context, provider)
        instruction = self._first_think_instruction(tools)
        observed = False

        for _ in range(context.max_iterations):
            async with aclosing(self._think(context, provider, instruction)) as chunks:
                async for chunk in chunks:
                    yield chunk
            thought = context.messages[-1]["content"]

            offered_tools = tools if self._wants_tool(thought, after_observation=observed) else []
            async with aclosing(self._act(context, provider, offered_tools)) as chunks:
                async for chunk in chunks:
                    yield chunk

            tool_calls = context.messages[-1].get("tool_calls", [])
            if not tool_calls:
                return

            async with aclosing(self._observe(context, tool_calls)) as chunks:
                async for chunk in chunks:
                    yield chunk
            instruction = OBSERVE_INSTRUCTION
            observed = True

        async with aclosing(self._request(context, provider, ANSWER_INSTRUCTION, [])) as chunks:
            async for chunk in chunks:
                yield chunk

    async def _think(
        self, context: ChatContext, provider: ProviderAdapter, instruction: str
    ) -> AsyncGenerator[ChatResponse]:
        """Make the model reason without answering, and show its reasoning as thinking.

        The step's own `done` chunk is dropped: the client never sees it end.
        """
        thought = ""

        async with aclosing(self._request(context, provider, instruction, [])) as chunks:
            async for chunk in chunks:
                if chunk.error is not None:
                    yield chunk
                if chunk.content is not None:
                    thought += chunk.content
                    yield self._build_thinking(chunk.content)
        yield self._build_thinking("\n")
        context.messages.append({"role": "assistant", "content": thought})

    async def _act(
        self, context: ChatContext, provider: ProviderAdapter, tools: list[dict]
    ) -> AsyncGenerator[ChatResponse]:
        """Let the model call one of `tools`, or make it write the final answer when there are none.

        Every chunk is forwarded, except the `done` one when the step called a tool: the exchange goes on, and the
        client stops streaming on the first `done` it gets. The tool calls come before the `done` chunk.
        """
        instruction = ACT_INSTRUCTION if tools else ANSWER_INSTRUCTION
        answer = ""
        tool_calls: list[dict] = []

        async with aclosing(self._request(context, provider, instruction, tools)) as chunks:
            async for chunk in chunks:
                if chunk.content is not None:
                    answer += chunk.content
                if chunk.tool_calls:
                    tool_calls.extend(chunk.tool_calls)
                if chunk.done and tool_calls:
                    continue
                yield chunk

        message: dict[str, object] = {"role": "assistant", "content": answer}
        if tool_calls:
            message["tool_calls"] = tool_calls
        context.messages.append(message)

    async def _observe(self, context: ChatContext, tool_calls: list[dict]) -> AsyncGenerator[ChatResponse]:
        """Run every tool the model called and add their results to `context.messages`, showing their names as thinking.

        `ToolManager.calls` asks for the confirmations, runs the approved tools in parallel, and turns every failure
        into an `error:` message for the model.
        """
        if context.tools is None:
            msg = "ReAct observed tool calls without a tool state"
            raise SensaiError(msg)

        for call in tool_calls:
            name = self._tool_manager.tool_name(call)
            yield self._build_thinking(f"-> Using {name}...\n")

        results = await self._tool_manager.calls(tool_calls, context.tools, ToolContext(out=context.out))
        context.messages.extend(results)

    async def _request(
        self, context: ChatContext, provider: ProviderAdapter, instruction: str, tools: list[dict]
    ) -> AsyncGenerator[ChatResponse]:
        """Stream the answer of the model to `context.messages` followed by `instruction`, offering it `tools`.

        `think` stays off: the loop is the thinking. The call is recorded and passed to `context.on_interaction`
        once it ends: `OK` when done, `ERROR` on an error chunk or a `SensaiError`, `INTERRUPTED` when closed or
        cancelled before its `done` chunk. Closed after it, the call is complete: the stream only had to report that
        it ran out. The error or the cancellation still propagates.
        """
        messages = [*context.messages, {"role": "user", "content": instruction}]
        payload: dict[str, object] = {"model": context.model, "messages": messages, "think": False}
        if tools:
            payload["tools"] = tools

        recorder = InteractionRecorder(model=context.model, request=messages)
        status, error = InteractionStatus.OK, None
        done = False
        try:
            async with aclosing(iterate_in_thread(partial(provider.chat, payload, stream=True))) as chunks:
                async for chunk in chunks:
                    recorder.add_chunk(chunk)
                    done = done or chunk.done
                    if chunk.error is not None:
                        status, error = InteractionStatus.ERROR, chunk.error
                    yield chunk
        except GeneratorExit, asyncio.CancelledError:
            if status is not InteractionStatus.ERROR and not done:
                status = InteractionStatus.INTERRUPTED
            raise
        except SensaiError as err:
            status, error = InteractionStatus.ERROR, str(err)
            raise
        finally:
            context.on_interaction(recorder.finish(status, error))

    async def _available_tools(self, context: ChatContext, provider: ProviderAdapter) -> list[dict]:
        """Return the tools to offer to the model, none when none is enabled or when it cannot call tools.

        The capabilities of the model are only looked up when there are tools to offer.
        """
        if context.tools is None:
            return []
        rules = self._tool_manager.get_tool_call_rules(context.tools)
        if not rules:
            return []

        capabilities = await asyncio.to_thread(provider.show, context.model)
        if not capabilities.tools:
            return []
        return rules

    @staticmethod
    def _wants_tool(thought: str, *, after_observation: bool) -> bool:
        """Read the decision ending `thought`: a tool is the default before any tool result, the answer after one.

        Only the last decision counts, since a model may quote the instruction ("End with 'Tool needed: yes' or
        'Tool needed: no'") or change its mind before deciding.
        """
        if after_observation:
            return _last_decision(ANSWER_DECISION, thought) == "no"
        return _last_decision(TOOL_DECISION, thought) != "no"

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


def _last_decision(decision: re.Pattern[str], thought: str) -> str | None:
    """Return the last "yes" or "no" `decision` found in `thought`, lowercased, or None when there is none."""
    answers = decision.findall(thought)
    return answers[-1].lower() if answers else None
