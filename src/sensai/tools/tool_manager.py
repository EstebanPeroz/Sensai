import json
import logging
from typing import TYPE_CHECKING, Any

from sensai.error import SensaiError
from sensai.tools.error import DuplicateToolError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sensai.tools.tool import Tool

logger = logging.getLogger(__name__)


class ToolManager:
    """Registry of the available tools, tracking which of them the model may call."""

    _tools: dict[str, Tool]
    _enabled: set[str]
    _unknown_disabled: frozenset[str]

    def __init__(self, tools: list[Tool] | None = None, disabled: Iterable[str] | None = None) -> None:
        """Register `tools`, enabling all of them but the ones named in `disabled`.

        Names in `disabled` matching no registered tool are ignored; `unknown_disabled()`
        reports them so the caller can warn about a stale or misspelled name.

        Raises:
            DuplicateToolError: If two tools share the same name.

        """
        self._tools = {}
        for tool in tools or []:
            if tool.name in self._tools:
                raise DuplicateToolError(tool.name)
            self._tools[tool.name] = tool
        requested = set(disabled or ())
        self._enabled = set(self._tools) - requested
        self._unknown_disabled = frozenset(requested - set(self._tools))

    def names(self) -> list[str]:
        """Return the name of every registered tool, enabled or not, sorted."""
        return sorted(self._tools)

    def unknown_disabled(self) -> list[str]:
        """Return the names given as disabled that match no registered tool, sorted."""
        return sorted(self._unknown_disabled)

    def is_enabled(self, name: str) -> bool:
        """Return whether `name` is a registered and enabled tool."""
        return name in self._enabled

    def enable(self, name: str) -> bool:
        """Enable the tool `name`, returning False if no tool has this name."""
        if name not in self._tools:
            return False
        self._enabled.add(name)
        return True

    def disable(self, name: str) -> bool:
        """Disable the tool `name`, returning False if no tool has this name."""
        if name not in self._tools:
            return False
        self._enabled.discard(name)
        return True

    def call(self, tool_call: dict[str, Any]) -> dict[str, str]:
        """Run the tool named in an LLM-style function-call payload and return its result as a tool message.

        A disabled tool is unreachable, like an unknown one, so a model still holding its schema in context
        cannot reach it. Every failure, a malformed call included, is returned as a message starting with
        `error:`, for the model to read.
        """
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if function is not None and not isinstance(function, dict):
            return self._message("unknown", "error: malformed tool call, function must be an object")
        name = (function or {}).get("name")
        if function is None or name is None:
            return self._message("unknown", "error: malformed tool call, missing function name")
        if not self.is_enabled(name):
            return self._message(name, f"error: unknown tool '{name}'")
        try:
            result = self._tools[name].run(**(function.get("arguments") or {}))
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

    def get_tool_call_rules(self) -> list[dict]:
        """Return the schema of every enabled tool, for the `tools` field of a chat payload."""
        return [self._tools[name].tool_rule for name in sorted(self._enabled)]

    @staticmethod
    def _message(tool_name: str, content: str) -> dict[str, str]:
        return {"role": "tool", "tool_name": tool_name, "content": content}
