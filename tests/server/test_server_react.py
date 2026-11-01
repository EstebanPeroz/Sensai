import asyncio
import threading
import warnings
from contextlib import aclosing
from pathlib import Path
from typing import Any

import pytest
from sensai_protocol.client_messages import Confirmation
from sensai_server.chat import chat
from sensai_server.chat.react import ANSWER_INSTRUCTION, ReAct
from sensai_server.events import EventBus
from sensai_server.llm.persona import PersonaManager
from sensai_server.llm.provider import ChatResponse, ProviderManager
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai_server.pipeline import ChatContext, ChatPipeline
from sensai_server.pipeline.modules import ModelStage
from sensai_server.session import Session
from sensai_server.tools.ask_user import AskUserTool
from sensai_server.tools.tool import Tool, ToolContext
from sensai_server.tools.tool_manager import ToolManager

from tests.server.conftest import MODEL, SETTINGS, db_url
from tests.server.fakes import FakeAdapter, RecordingOutbound, call, chunk, tool_call_chunk

TIMEOUT = 2
ANSWER = "Bonjour !"
THOUGHT = "I need a recent fact.\nTool needed: yes"
SEARCHED = "The page is empty.\nAnswer found: no"


class SearchTool(Tool[str]):
    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search the web."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def run(self, ctx: ToolContext, **kwargs: object) -> str:
        del ctx, kwargs
        return "a result"


class BlockingTool(SearchTool):
    """Runs until cancelled, setting `started` once running and `cancelled` once cancelled."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def run(self, ctx: ToolContext, **kwargs: object) -> str:
        del ctx, kwargs
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()
        return "never"


class ConfirmingOutbound(RecordingOutbound):
    """Answers every confirmation with `answer`, and every question with `reply`, recording what was asked."""

    def __init__(self, answer: Confirmation = Confirmation.DENY, reply: str | None = None) -> None:
        super().__init__()
        self._answer = answer
        self._reply = reply
        self.confirmations: list[str] = []
        self.questions: list[str] = []

    async def ask_confirmation(self, tool_name: str, arguments: dict[str, Any]) -> Confirmation:
        del arguments
        self.confirmations.append(tool_name)
        return self._answer

    async def ask_user(self, question: str) -> str | None:
        self.questions.append(question)
        return self._reply


def _react(manager: ToolManager) -> ReAct:
    return ReAct(ProviderManager(SETTINGS.llm), manager)


def _context(
    manager: ToolManager | None,
    *,
    max_iterations: int = 1,
    prompt: str = "Bonjour",
    out: RecordingOutbound | None = None,
) -> ChatContext:
    return ChatContext(
        message=prompt,
        out=out or RecordingOutbound(),
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        tools=manager.new_state() if manager is not None else None,
        max_iterations=max_iterations,
    )


async def _run(manager: ToolManager, context: ChatContext) -> list[ChatResponse]:
    async with aclosing(_react(manager).generate_answer(context)) as chunks:
        return [item async for item in chunks]


def _answer(chunks: list[ChatResponse]) -> str:
    return "".join(item.content or "" for item in chunks)


def _script_tool_round(*calls: dict, thought: str = THOUGHT) -> list[list[dict]]:
    """A Think step deciding `thought`, then an Act step calling `calls`."""
    return [[chunk(thought, done=True)], [tool_call_chunk(*calls), chunk("", done=True)]]


@pytest.fixture(autouse=True)
def answers() -> None:
    """Every call not scripted answers `ANSWER`, by a model able to call tools."""
    FakeAdapter.replies = [chunk(ANSWER, done=True)]
    FakeAdapter.tools = True


class TestEffortOff:
    async def test_answers_with_a_single_request_without_tools(self) -> None:
        manager = ToolManager([SearchTool()])
        chunks = await _run(manager, _context(manager, max_iterations=0))

        assert _answer(chunks) == ANSWER
        assert len(FakeAdapter.payloads) == 1
        assert "tools" not in FakeAdapter.payloads[0]
        assert FakeAdapter.payloads[0]["messages"][-1] == {"role": "user", "content": ANSWER_INSTRUCTION}

    async def test_does_not_look_up_the_model_capabilities(self) -> None:
        manager = ToolManager([SearchTool()])
        await _run(manager, _context(manager, max_iterations=0))

        assert FakeAdapter.shown == []

    async def test_capabilities_are_looked_up_when_thinking_is_allowed(self) -> None:
        manager = ToolManager([SearchTool()])
        await _run(manager, _context(manager))

        assert FakeAdapter.shown == [MODEL]


class TestWithoutTools:
    @pytest.mark.parametrize("manager", [ToolManager(), ToolManager([SearchTool()], disabled=["search"])])
    async def test_no_enabled_tool_does_not_look_up_the_model_capabilities(self, manager: ToolManager) -> None:
        chunks = await _run(manager, _context(manager))

        assert FakeAdapter.shown == []
        assert _answer(chunks) == ANSWER
        assert all("tools" not in payload for payload in FakeAdapter.payloads)

    async def test_no_tool_state_does_not_look_up_the_model_capabilities(self) -> None:
        manager = ToolManager([SearchTool()])
        chunks = await _run(manager, _context(None))

        assert FakeAdapter.shown == []
        assert _answer(chunks) == ANSWER
        assert all("tools" not in payload for payload in FakeAdapter.payloads)

    async def test_model_without_tool_capability_is_offered_none(self) -> None:
        FakeAdapter.tools = False
        manager = ToolManager([SearchTool()])
        await _run(manager, _context(manager))

        assert FakeAdapter.shown == [MODEL]
        assert all("tools" not in payload for payload in FakeAdapter.payloads)


class TestMalformedToolCall:
    async def test_is_reported_to_the_model_instead_of_crashing(self) -> None:
        FakeAdapter.script = _script_tool_round({"function": "search"})
        manager = ToolManager([SearchTool()])

        chunks = await _run(manager, _context(manager, prompt="Find it"))

        assert {
            "role": "tool",
            "tool_name": "unknown",
            "content": "error: malformed tool call, function must be an object",
        } in FakeAdapter.payloads[-1]["messages"]
        assert "-> Using unknown...\n" in [item.thinking for item in chunks]


class TestConfirmation:
    async def _run_gated(self, answer: Confirmation) -> tuple[ConfirmingOutbound, list[ChatResponse]]:
        FakeAdapter.script = _script_tool_round(call("search"))
        manager = ToolManager([SearchTool()], confirm=("search",))
        out = ConfirmingOutbound(answer)
        chunks = await _run(manager, _context(manager, prompt="Find it", out=out))
        return out, chunks

    async def test_refusal_is_sent_to_the_model_and_the_loop_still_answers(self) -> None:
        out, chunks = await self._run_gated(Confirmation.DENY)

        assert out.confirmations == ["search"]
        assert {
            "role": "tool",
            "tool_name": "search",
            "content": "error: the user declined to run 'search'; do not call it again, answer without it",
        } in FakeAdapter.payloads[-1]["messages"]
        assert _answer(chunks) == ANSWER

    async def test_allowed_call_sends_the_tool_result_to_the_model(self) -> None:
        out, _ = await self._run_gated(Confirmation.ALLOW)

        assert out.confirmations == ["search"]
        assert {"role": "tool", "tool_name": "search", "content": "a result"} in FakeAdapter.payloads[-1]["messages"]


class TestAskUser:
    async def test_answer_of_the_user_reaches_the_final_request(self) -> None:
        FakeAdapter.script = _script_tool_round(call("ask_user", {"question": "Which quarter?"}))
        manager = ToolManager([AskUserTool()])
        out = ConfirmingOutbound(reply="Q2 2026")

        await _run(manager, _context(manager, prompt="How did we do?", out=out))

        assert out.questions == ["Which quarter?"]
        assert {"role": "tool", "tool_name": "ask_user", "content": "Q2 2026"} in FakeAdapter.payloads[-1]["messages"]

    async def test_is_listed_in_the_first_think_instruction(self) -> None:
        manager = ToolManager([AskUserTool()])

        await _run(manager, _context(manager, prompt="How did we do?"))

        assert "- ask_user: Ask the user one short question" in FakeAdapter.payloads[0]["messages"][-1]["content"]


async def _tools_offered_after(thought: str) -> bool:
    FakeAdapter.script = [[chunk(thought, done=True)]]
    manager = ToolManager([SearchTool()])
    await _run(manager, _context(manager))
    return "tools" in FakeAdapter.payloads[1]


class TestToolDecision:
    async def test_yes_offers_the_tools(self) -> None:
        assert await _tools_offered_after("I need a recent fact.\nTool needed: yes")

    async def test_no_does_not_offer_the_tools(self) -> None:
        assert not await _tools_offered_after("It is a greeting.\nTool needed: no")

    async def test_no_decision_offers_the_tools(self) -> None:
        assert await _tools_offered_after("I am not sure.")

    async def test_quoting_the_instruction_does_not_count_as_a_refusal(self) -> None:
        thought = "I must end with 'Tool needed: yes' or 'Tool needed: no'. I lack the data.\nTool needed: yes"

        assert await _tools_offered_after(thought)

    async def test_only_the_last_decision_counts(self) -> None:
        assert not await _tools_offered_after("Tool needed: yes? No, it is arithmetic.\nTool needed: no")

    async def test_reads_a_decision_in_markdown(self) -> None:
        assert not await _tools_offered_after("It is a greeting.\n**Tool needed:** No")


class TestAnswerDecision:
    def test_searches_again_only_when_the_last_decision_is_no(self) -> None:
        thought = "End with 'Answer found: yes' or 'Answer found: no'. The page has the figure.\nAnswer found: yes"

        assert not ReAct._wants_tool(thought, after_observation=True)  # noqa: SLF001
        assert ReAct._wants_tool("The page is empty.\nAnswer found: no", after_observation=True)  # noqa: SLF001

    def test_answers_without_a_decision(self) -> None:
        assert not ReAct._wants_tool("The page has the figure.", after_observation=True)  # noqa: SLF001


class TestOneDone:
    async def test_a_run_with_a_tool_call_sends_a_single_done_chunk_last(self) -> None:
        FakeAdapter.script = _script_tool_round(call("search"))
        manager = ToolManager([SearchTool()])

        chunks = await _run(manager, _context(manager))

        assert [item.done for item in chunks].count(True) == 1
        assert chunks[-1].done
        assert _answer(chunks) == ANSWER

    async def test_think_shows_its_reasoning_as_thinking(self) -> None:
        FakeAdapter.script = _script_tool_round(call("search"))
        manager = ToolManager([SearchTool()])

        chunks = await _run(manager, _context(manager))

        thinking = "".join(item.thinking or "" for item in chunks)
        assert thinking == f"{THOUGHT}\n-> Using search...\n"


class TestIsolation:
    async def test_contexts_run_at_once_offer_only_their_own_tools(self) -> None:
        manager = ToolManager([SearchTool(), AskUserTool()])
        first, second = _context(manager, prompt="first"), _context(manager, prompt="second")
        assert first.tools is not None
        assert second.tools is not None
        manager.disable(first.tools, "ask_user")
        manager.disable(second.tools, "search")
        react = _react(manager)

        async def run(context: ChatContext) -> None:
            async with aclosing(react.generate_answer(context)) as chunks:
                async for _ in chunks:
                    await asyncio.sleep(0)

        await asyncio.gather(run(first), run(second))

        offered: dict[str, set[str]] = {}
        for payload in FakeAdapter.payloads:
            if "tools" in payload:
                prompt = payload["messages"][0]["content"]
                offered[prompt] = {tool["function"]["name"] for tool in payload["tools"]}
        assert offered == {"first": {"search"}, "second": {"ask_user"}}


class TestEffortPerExchange:
    @pytest.mark.parametrize("steps", [1, 2, 3])
    async def test_max_iterations_bounds_the_loop_and_forces_the_answer(self, steps: int) -> None:
        FakeAdapter.script = [
            stream
            for step in range(steps)
            for stream in _script_tool_round(call("search"), thought=SEARCHED if step else THOUGHT)
        ]
        manager = ToolManager([SearchTool()])

        chunks = await _run(manager, _context(manager, max_iterations=steps))

        assert len(FakeAdapter.payloads) == 2 * steps + 1
        last = FakeAdapter.payloads[-1]
        assert last["messages"][-1] == {"role": "user", "content": ANSWER_INSTRUCTION}
        assert "tools" not in last
        assert _answer(chunks) == ANSWER


class TestHistoryUntouched:
    async def test_saved_history_holds_only_the_prompt_and_the_answer(
        self, tmp_path: Path, persona_manager: PersonaManager
    ) -> None:
        FakeAdapter.script = _script_tool_round(call("search"))
        pm = ProviderManager(SETTINGS.llm)
        manager = ToolManager([SearchTool()])
        session = Session(RecordingOutbound(), pm, persona_manager, manager, effort="low")
        session.conv.model = MODEL
        pipeline = ChatPipeline(ModelStage(ReAct(pm, manager)))

        await chat(session, "Find it", pipeline, SqlAlchemy(db_url(tmp_path)), EventBus())

        assert {"role": "tool", "tool_name": "search", "content": "a result"} in FakeAdapter.payloads[-1]["messages"]
        assert session.conv.history.to_json() == [
            {"role": "user", "content": "Find it"},
            {"role": "assistant", "content": ANSWER},
        ]


class TestCancel:
    async def test_cancelling_during_a_tool_call_cancels_the_tool(self) -> None:
        FakeAdapter.script = _script_tool_round(call("search"))
        tool = BlockingTool()
        manager = ToolManager([tool])

        task = asyncio.create_task(_run(manager, _context(manager)))
        await asyncio.wait_for(tool.started.wait(), TIMEOUT)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert tool.cancelled.is_set()

    async def test_closing_during_a_step_closes_the_provider_stream(self) -> None:
        FakeAdapter.block = threading.Event()
        FakeAdapter.replies = [chunk("a"), chunk("b", done=True)]
        manager = ToolManager([SearchTool()])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            async with aclosing(_react(manager).generate_answer(_context(manager))) as chunks:
                async for item in chunks:
                    assert item.thinking == "a"
                    break
            assert not FakeAdapter.closed.is_set()
            # The thread is still inside `next()`: the close runs once it returns.
            FakeAdapter.block.set()
            assert await asyncio.to_thread(FakeAdapter.closed.wait, TIMEOUT)
        assert len(FakeAdapter.payloads) == 1
