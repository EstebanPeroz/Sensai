from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from benchmarks.models.runner import NO_RESULT, Benchmark, ScriptedTool
from sensai.llm.responses import ChatResponse, ShowResponse
from sensai.tools.ask_user import AskUserTool
from sensai.tools.web.backend.search import SearchChain
from sensai.tools.web.search import WebSearchTool

if TYPE_CHECKING:
    from collections.abc import Iterator

MODEL = "tiny:1b"


class FakeProvider:
    """Thinks it needs a tool, calls `web_search` when it is offered, then answers with what the tool returned."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def current_model(self) -> str:
        return MODEL

    def show(self, model: str) -> ShowResponse:
        return ShowResponse(model, tools=True)

    def chat(self, payload: dict, *, stream: bool) -> Iterator[ChatResponse]:  # noqa: ARG002
        self.payloads.append(payload)
        instruction = payload["messages"][-1]["content"]
        if instruction.startswith("Do not answer yet"):
            message = {"content": "Tool needed: yes. Answer found: yes."}
        elif "tools" in payload:
            call = {"function": {"name": "web_search", "arguments": {"query": "tour de france 2026"}}}
            message = {"content": "", "tool_calls": [call]}
        else:
            tool_results = [m["content"] for m in payload["messages"] if m.get("role") == "tool"]
            message = {"content": tool_results[-1] if tool_results else "I am fine."}
        yield ChatResponse({"done": True, "message": {"role": "assistant", **message}, "prompt_eval_count": 50})


def benchmark(cases: dict) -> Benchmark:
    manager = MagicMock()
    manager.provider.return_value = FakeProvider()
    tools = [WebSearchTool(SearchChain([])), AskUserTool(lambda _question: None)]
    return Benchmark(manager, tools, cases, context_window=None, progress=lambda _line: None)


def tool_case() -> dict:
    return {"id": "case", "prompt": "Who won?", "tool": "web_search", "result": "Lukas Brenner won.", "expected": []}


class TestTools:
    def test_scores_the_expected_tool_and_the_answer_built_on_its_result(self) -> None:
        case = tool_case() | {"expected": ["brenner"]}

        [result] = benchmark({"tools": [case]}).tools()

        assert result["called"] == ["web_search"]
        assert result["right_tool"]
        assert result["valid_arguments"]
        assert result["answered"]
        assert result["passed"]
        assert result["prompt_tokens"] == 50

    def test_calling_a_tool_when_none_is_needed_fails(self) -> None:
        case = tool_case() | {"tool": "", "expected": []}

        [result] = benchmark({"tools": [case]}).tools()

        assert not result["right_tool"]
        assert not result["passed"]

    def test_another_tool_than_the_expected_one_finds_nothing(self) -> None:
        case = tool_case() | {"tool": "ask_user", "expected": ["brenner"]}

        [result] = benchmark({"tools": [case]}).tools()

        assert result["answer"] == NO_RESULT
        assert not result["right_tool"]
        assert not result["answered"]


def test_scripted_tool_shows_the_real_tool_and_returns_its_result() -> None:
    real = WebSearchTool(SearchChain([]))
    tool = ScriptedTool(real)
    tool.result = "scripted"

    assert (tool.name, tool.description, tool.parameters) == (real.name, real.description, real.parameters)
    assert tool.run(query="anything") == "scripted"
