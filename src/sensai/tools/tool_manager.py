from typing import TYPE_CHECKING, Any

from sensai.tools.error import DuplicateToolError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sensai.tools.tool import Tool


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

    def call(self, tool_call: dict[str, Any]) -> dict | None:
        """Call the tool named in an LLM-style function-call payload.

        Return an empty dict if the payload names no tool, and None if no *enabled*
        tool has this name: a disabled tool is unreachable, like an unknown one, so a
        model still holding its schema in context cannot reach it.
        """
        function = tool_call.get("function")
        if function is None:
            return {}
        tool_name = function.get("name")
        if tool_name is None:
            return {}
        if not self.is_enabled(tool_name):
            return None
        arguments = function.get("arguments", {})
        return {"role": "tool", "tool_name": tool_name, "content": self._tools[tool_name].run(**arguments)}

    def get_tool_call_rules(self) -> list[dict]:
        """Return the schema of every enabled tool, for the `tools` field of a chat payload."""
        return [self._tools[name].tool_rule for name in sorted(self._enabled)]
