from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from sensai_protocol.client_messages import Confirmation
from sensai_protocol.error import SensaiError

from sensai_server.tools.error import DuplicateToolError
from sensai_server.tools.state import ToolState

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sensai_server.tools.tool import Tool, ToolContext

logger = logging.getLogger(__name__)


class ToolManager:
    """Registry of the available tools, shared by every session.

    It keeps no per-client state: each session holds its own `ToolState`, built by `new_state`.
    """

    _tools: dict[str, Tool]
    _disabled: frozenset[str]
    _unknown_disabled: frozenset[str]
    _confirm: frozenset[str]
    _unknown_confirm: frozenset[str]

    def __init__(
        self,
        tools: list[Tool] | None = None,
        disabled: Iterable[str] | None = None,
        *,
        confirm: Iterable[str] | None = None,
    ) -> None:
        """Register `tools`; a new session enables all of them but the ones named in `disabled`.

        The tools named in `confirm`, and those requiring it themselves, run only once the user allows them.

        Names in `disabled` or `confirm` matching no registered tool are ignored; `unknown_disabled()`
        and `unknown_confirm()` report them so the caller can warn about a stale or misspelled name.

        Raises:
            DuplicateToolError: If two tools share the same name.

        """
        self._tools = {}
        for tool in tools or []:
            if tool.name in self._tools:
                raise DuplicateToolError(tool.name)
            self._tools[tool.name] = tool
        requested = frozenset(disabled or ())
        self._disabled = requested & set(self._tools)
        self._unknown_disabled = requested - set(self._tools)
        self._confirm = frozenset(confirm or ())
        self._unknown_confirm = self._confirm - set(self._tools)

    def names(self) -> list[str]:
        """Return the name of every registered tool, enabled or not, sorted."""
        return sorted(self._tools)

    def unknown_disabled(self) -> list[str]:
        """Return the names given as disabled that match no registered tool, sorted."""
        return sorted(self._unknown_disabled)

    def unknown_confirm(self) -> list[str]:
        """Return the names given as needing confirmation that match no registered tool, sorted."""
        return sorted(self._unknown_confirm)

    def new_state(self) -> ToolState:
        """Return the tool state of a new session: every tool enabled but the disabled ones, none always allowed."""
        return ToolState(enabled=set(self._tools) - self._disabled)

    def is_enabled(self, state: ToolState, name: str) -> bool:
        """Return whether `name` is a registered tool enabled in `state`."""
        return name in self._tools and name in state.enabled

    def needs_confirmation(self, state: ToolState, name: str) -> bool:
        """Return whether calling the tool `name` still needs the user's approval in the session of `state`."""
        tool = self._tools.get(name)
        if tool is None or name in state.always_allowed:
            return False
        return tool.requires_confirmation or name in self._confirm

    def enable(self, state: ToolState, name: str) -> bool:
        """Enable the tool `name` in `state`, returning False if no tool has this name."""
        if name not in self._tools:
            return False
        state.enabled.add(name)
        return True

    def disable(self, state: ToolState, name: str) -> bool:
        """Disable the tool `name` in `state`, returning False if no tool has this name."""
        if name not in self._tools:
            return False
        state.enabled.discard(name)
        return True

    def get_tool_call_rules(self, state: ToolState) -> list[dict]:
        """Return the schema of every tool enabled in `state`, for the `tools` field of a chat payload."""
        return [self._tools[name].tool_rule for name in sorted(state.enabled) if name in self._tools]

    async def call(self, tool_call: dict[str, Any], state: ToolState, ctx: ToolContext) -> dict[str, str]:
        """Run a single LLM-style function call, as `calls` does."""
        return (await self.calls([tool_call], state, ctx))[0]

    async def calls(self, tool_calls: list[dict[str, Any]], state: ToolState, ctx: ToolContext) -> list[dict[str, str]]:
        """Run the tools named in LLM-style function-call payloads and return one tool message per call, in order.

        A disabled tool is unreachable, like an unknown one, so a model still holding its schema in context
        cannot reach it. The calls needing confirmation are asked to the client of `ctx` one after another, in
        call order; the approved calls then run in parallel. Every failure, a malformed call or a refusal included,
        is returned as a message starting with `error:`, for the model to read, and never stops the other calls.
        Cancelling this coroutine cancels every running tool.
        """
        results: list[dict[str, str] | None] = [None] * len(tool_calls)
        approved: list[tuple[int, str, Any]] = []
        for index, tool_call in enumerate(tool_calls):
            checked = self._check(tool_call, state)
            if isinstance(checked, dict):
                results[index] = checked
                continue
            name, arguments = checked
            if not await self._approve(state, name, arguments, ctx):
                refusal = f"error: the user declined to run '{name}'; do not call it again, answer without it"
                results[index] = self._message(name, refusal)
                continue
            approved.append((index, name, arguments))
        ran = await asyncio.gather(*(self._run(name, arguments, ctx) for _, name, arguments in approved))
        for (index, _, _), message in zip(approved, ran, strict=True):
            results[index] = message
        return [result for result in results if result is not None]

    def _check(self, tool_call: object, state: ToolState) -> tuple[str, Any] | dict[str, str]:
        """Return the name and arguments of a well-formed call to an enabled tool, or the error as a tool message."""
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if function is not None and not isinstance(function, dict):
            return self._message("unknown", "error: malformed tool call, function must be an object")
        name = (function or {}).get("name")
        if function is None or name is None:
            return self._message("unknown", "error: malformed tool call, missing function name")
        if not self.is_enabled(state, name):
            return self._message(name, f"error: unknown tool '{name}'")
        return name, function.get("arguments") or {}

    async def _approve(self, state: ToolState, name: str, arguments: object, ctx: ToolContext) -> bool:
        """Return whether the tool `name` may run, asking the client of `ctx` when it needs confirmation."""
        if not self.needs_confirmation(state, name):
            return True
        answer = await ctx.out.ask_confirmation(name, arguments if isinstance(arguments, dict) else {})
        if answer is Confirmation.ALWAYS:
            state.always_allowed.add(name)
        return answer in {Confirmation.ALLOW, Confirmation.ALWAYS}

    async def _run(self, name: str, arguments: Any, ctx: ToolContext) -> dict[str, str]:  # noqa: ANN401 - unchecked model output
        """Run the tool `name` with `arguments` and return its result, or the failure, as a tool message."""
        try:
            result = await self._tools[name].run(ctx, **arguments)
            content = result if isinstance(result, str) else json.dumps(result, default=str)
        except (SensaiError, TypeError) as err:
            return self._message(name, f"error: {err}")
        except Exception as err:
            logger.exception("Tool '%s' failed unexpectedly", name)
            return self._message(name, f"error: {type(err).__name__}: {err}")
        return self._message(name, content)

    @staticmethod
    def tool_name(tool_call: object) -> str:
        """Return the name of the tool an LLM-style call asks for, or `unknown` when the call is malformed."""
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if not isinstance(function, dict) or function.get("name") is None:
            return "unknown"
        return str(function["name"])

    @staticmethod
    def _message(tool_name: str, content: str) -> dict[str, str]:
        return {"role": "tool", "tool_name": tool_name, "content": content}
