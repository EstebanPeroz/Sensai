"""Runs of the model benchmark, each one against the model the provider manager has loaded."""

from __future__ import annotations

import statistics
import time
from contextlib import closing
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from benchmarks.models import checks
from sensai.error import SensaiError
from sensai.llm.thinking.react import ReAct
from sensai.pipeline.modules.compression import Compressor
from sensai.tools.tool import Tool
from sensai.tools.tool_manager import ToolManager

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from sensai.llm.provider_manager import ProviderManager
    from sensai.memory.summary import Summary
    from sensai.monitoring.interaction import Interaction

NO_RESULT = "No relevant result found."


class BenchmarkError(SensaiError):
    """Raised when the model fails a call the benchmark cannot score without."""


@dataclass
class Reply:
    """What the model answered to one question through the ReAct loop, and what it took."""

    text: str
    seconds: float
    model_calls: int
    prompt_tokens: int | None
    """Largest prompt of the calls, the whole context the model read at once."""
    tool_calls: list[tuple[str, Any]] = field(default_factory=list)
    """Name and arguments of every tool call the model made, valid or not."""
    error: str | None = None


class ScriptedTool(Tool[str]):
    """Stand-in for a real tool: the name, description and parameters the model reads, and a scripted result."""

    def __init__(self, tool: Tool) -> None:
        """Show the model what `tool` shows it, answering with `result` instead of running it."""
        self._tool = tool
        self.result = NO_RESULT

    @property
    def name(self) -> str:
        """Name of the real tool."""
        return self._tool.name

    @property
    def description(self) -> str:
        """Description of the real tool."""
        return self._tool.description

    @property
    def parameters(self) -> dict:
        """Parameters of the real tool."""
        return self._tool.parameters

    def run(self, **_kwargs: object) -> str:
        """Return the scripted result, whatever the arguments."""
        return self.result


class InMemorySummaries:
    """Summary storage that lasts as long as the benchmark."""

    def __init__(self) -> None:
        """Start without any summary."""
        self._summaries: dict[UUID, Summary] = {}

    def get(self, branch_id: UUID) -> Summary | None:
        """Return the summary of the branch `branch_id`, None when it has none."""
        return self._summaries.get(branch_id)

    def save(self, branch_id: UUID, summary: Summary) -> None:
        """Make `summary` the summary of the branch `branch_id`."""
        self._summaries[branch_id] = summary


class Benchmark:
    """Ask the cases of the benchmark to the current model of `provider_manager`, through Sensai's own ReAct loop."""

    def __init__(
        self,
        provider_manager: ProviderManager,
        tools: list[Tool],
        cases: dict[str, Any],
        *,
        context_window: int | None,
        progress: Callable[[str], object],
    ) -> None:
        """Offer the model scripted copies of `tools`, within the `context_window` of the chat, telling `progress`."""
        self._provider_manager = provider_manager
        self._cases = cases
        self._context_window = context_window
        self._progress = progress
        self._tools = {tool.name: ScriptedTool(tool) for tool in tools}
        self._plain = ReAct(provider_manager, ToolManager([]), context_window=context_window)
        self._tooled = ReAct(provider_manager, ToolManager(list(self._tools.values())), context_window=context_window)

    def latency(self) -> dict[str, Any]:
        """Time a cold call, then the median of warm ones, alone and after a long context.

        Every warm call starts differently, so that the provider cannot reuse the prompt it read for the previous one.
        """
        case = self._cases["latency"]
        provider = self._provider_manager.provider()
        model = provider.current_model()
        provider.unload(model)
        cold = self._sample(case["question"], model)
        # Unloading leaves the provider without a current model; the cold call has just loaded it back.
        provider.load(model)
        self._progress(f"[latency] cold: {cold['load_s']} s to load")
        context = case["filler"] * case["filler_repeat"]
        short = [self._sample(f"Run {run}. {case['question']}") for run in range(1, case["runs"] + 1)]
        long = [self._sample(f"Run {run}.\n\n{context}\n\n{case['question']}") for run in range(1, case["runs"] + 1)]
        self._progress("[latency] warm runs done")
        return {"cold": cold, "short": _median(short), "long": _median(long)}

    def quality(self) -> list[dict[str, Any]]:
        """Ask each question with a single right answer, without tools."""
        results = []
        for case in self._cases["quality"]:
            reply = self._ask(self._plain, [_user(case["prompt"])])
            passed = reply.error is None and checks.matches(reply.text, case["expected"])
            results.append(self._result("quality", case["id"], reply, passed=passed))
        return results

    def tools(self) -> list[dict[str, Any]]:
        """Ask each question with the tools on offer, scoring the tool called first, its arguments and the answer.

        Only the tool the case expects returns its scripted result; any other one finds nothing.
        """
        results = []
        for case in self._cases["tools"]:
            expected_tool = case["tool"] or None
            for name, tool in self._tools.items():
                tool.result = case.get("result", NO_RESULT) if name == expected_tool else NO_RESULT
            reply = self._ask(self._tooled, [_user(case["prompt"])])
            called = [name for name, _ in reply.tool_calls]
            right_tool = called[:1] == [expected_tool] if expected_tool else not called
            valid = all(
                name in self._tools and checks.valid_arguments(arguments, self._tools[name].parameters)
                for name, arguments in reply.tool_calls
            )
            answered = reply.error is None and checks.matches(reply.text, case["expected"])
            result = self._result("tools", case["id"], reply, passed=right_tool and valid and answered)
            result |= {
                "expected_tool": expected_tool,
                "called": called,
                "right_tool": right_tool,
                "valid_arguments": valid,
                "answered": answered,
            }
            results.append(result)
        return results

    def instructions(self) -> list[dict[str, Any]]:
        """Ask each question with constraints on the answer, which must pass every one of its checks."""
        results = []
        for case in self._cases["instructions"]:
            reply = self._ask(self._plain, [_user(case["prompt"])])
            passed = reply.error is None and all(checks.passes(reply.text, check) for check in case["checks"])
            results.append(self._result("instructions", case["id"], reply, passed=passed))
        return results

    def memory(self) -> dict[str, Any]:
        """Ask back the facts of a long conversation, from the whole of it, then from its compressed form."""
        case = self._cases["memory"]
        history: list[dict] = []
        for fact in case["facts"]:
            history += [_user(fact), {"role": "assistant", "content": case["fact_reply"]}]
        for exchange in case["filler"]:
            history += [_user(exchange["question"]), {"role": "assistant", "content": exchange["answer"]}]

        full = self._recall("memory/full", history, case["questions"])
        compressor = Compressor(
            self._provider_manager, InMemorySummaries(), case["keep_recent"], context_window=self._context_window
        )
        branch = uuid4()
        start = time.perf_counter()
        try:
            compression = compressor.compress(branch, history)
        except SensaiError as err:
            self._progress(f"[memory] compression failed: {err}")
            return {"full": full, "compressed": [], "compression": {"error": str(err)}}
        seconds = time.perf_counter() - start
        shortened = compressor.apply(branch, history)
        compressed = self._recall("memory/compressed", shortened, case["questions"])
        summary = shortened[0]["content"] if compression is not None else None
        stats = asdict(compression) if compression is not None else {}
        return {
            "full": full,
            "compressed": compressed,
            "compression": stats | {"seconds": round(seconds, 2), "summary": summary},
        }

    def _recall(self, category: str, history: list[dict], questions: list[dict]) -> list[dict[str, Any]]:
        """Ask each of `questions` after `history`, on its own."""
        results = []
        for number, question in enumerate(questions, start=1):
            reply = self._ask(self._plain, [*history, _user(question["prompt"])])
            passed = reply.error is None and checks.matches(reply.text, question["expected"])
            results.append(self._result(category, f"q{number}", reply, passed=passed))
        return results

    def _ask(self, react: ReAct, messages: list[dict]) -> Reply:
        """Stream the answer of `react` to `messages`, timing it and keeping every call it made to the model."""
        interactions: list[Interaction] = []
        text, error = "", None
        start = time.perf_counter()
        try:
            with closing(react.generate_answer(list(messages), interactions.append)) as chunks:
                for chunk in chunks:
                    if chunk.error is not None:
                        error = chunk.error
                        break
                    text += chunk.content or ""
        except SensaiError as err:
            error = str(err)
        seconds = time.perf_counter() - start
        tool_calls = [
            (ToolManager.tool_name(call), (call.get("function") or {}).get("arguments"))
            for interaction in interactions
            for call in interaction.tool_calls
        ]
        prompts = [i.metrics.prompt_tokens for i in interactions if i.metrics.prompt_tokens is not None]
        return Reply(text.strip(), seconds, len(interactions), max(prompts, default=None), tool_calls, error)

    def _result(self, category: str, case_id: str, reply: Reply, *, passed: bool) -> dict[str, Any]:
        """Report the outcome of one case, and keep it for the results file."""
        mark = "ok  " if passed else "FAIL"
        self._progress(f"[{category}] {mark} {case_id} ({reply.seconds:.1f} s, {reply.model_calls} calls)")
        return {
            "id": case_id,
            "passed": passed,
            "seconds": round(reply.seconds, 2),
            "model_calls": reply.model_calls,
            "prompt_tokens": reply.prompt_tokens,
            "answer": reply.text,
            "error": reply.error,
        }

    def _sample(self, prompt: str, model: str | None = None) -> dict[str, float | int | None]:
        """Send `prompt` straight to `model`, the current one by default, outside of the ReAct loop, and time it.

        Raises:
            BenchmarkError: If the model answers with an error.

        """
        provider = self._provider_manager.provider()
        model = model or provider.current_model()
        payload: dict[str, Any] = {"model": model, "messages": [_user(prompt)], "think": False}
        if self._context_window is not None:
            payload["options"] = {"num_ctx": self._context_window}
        first_token, metrics = None, None
        start = time.perf_counter()
        for chunk in provider.chat(payload, stream=True):
            if chunk.error is not None:
                raise BenchmarkError(chunk.error)
            if first_token is None and chunk.content:
                first_token = time.perf_counter() - start
            metrics = chunk.metrics or metrics
        total = time.perf_counter() - start
        if metrics is None:
            msg = "the model did not report its metrics"
            raise BenchmarkError(msg)
        return {
            "first_token_s": _round(first_token),
            "total_s": _round(total),
            "load_s": _round(_seconds(metrics.load_duration_ms)),
            "prompt_tokens": metrics.prompt_tokens,
            "prompt_tokens_per_s": _round(_rate(metrics.prompt_tokens, metrics.prompt_eval_duration_ms)),
            "completion_tokens": metrics.completion_tokens,
            "completion_tokens_per_s": _round(_rate(metrics.completion_tokens, metrics.eval_duration_ms)),
        }


def _user(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


def _seconds(milliseconds: float | None) -> float | None:
    return None if milliseconds is None else milliseconds / 1000


def _rate(tokens: int | None, milliseconds: float | None) -> float | None:
    """Tokens per second, None when the provider did not report them or the duration is zero."""
    if tokens is None or not milliseconds:
        return None
    return tokens / (milliseconds / 1000)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def _median(samples: list[dict[str, float | int | None]]) -> dict[str, float | None]:
    """Median of each measure across `samples`, None for a measure none of them has."""
    keys = samples[0].keys() if samples else []
    medians: dict[str, float | None] = {}
    for key in keys:
        values = [value for sample in samples if (value := sample[key]) is not None]
        medians[key] = _round(statistics.median(values)) if values else None
    return medians
