from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sensai.tools.tool import Tool


class ToolManager:
    """Registry of the available tools, tracking which of them the model may call."""

    _tools: dict[str, Tool]
    _enabled: set[str]

    def __init__(self, tools: list[Tool] | None = None, disabled: Iterable[str] | None = None) -> None:
        """Register `tools`, enabling all of them but the ones named in `disabled`."""
        self._tools = {tool.name: tool for tool in tools or []}
        self._enabled = set(self._tools) - set(disabled or ())

    def names(self) -> list[str]:
        """Return the name of every registered tool, enabled or not, sorted."""
        return sorted(self._tools)

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
        """Call the tool named function-call payload."""
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
