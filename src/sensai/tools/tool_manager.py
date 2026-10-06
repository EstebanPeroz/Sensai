from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sensai.tools.tool import Tool


class ToolManager:
    """Tmp."""

    _tools: list[Tool]

    def __init__(self, tools: list[Tool] | None = None) -> None:
        """Tmp."""
        self._tools: list[Tool] = tools or []

    def call(self, tool_call: dict[str, Any]) -> dict | None:
        """Call the tool named in an LLM-style function-call payload."""
        function = tool_call.get("function")
        if function is None:
            return {}
        tool_name = function.get("name")
        arguments = function.get("arguments", {})
        if tool_name is None:
            return {}

        for tool in self._tools:
            if tool_name == tool.name:
                return {"role": "tool", "tool_name": tool_name, "content": tool.run(**arguments)}
        return None

    def get_tool_call_rules(self) -> list[dict]:
        """Tmp."""
        return [tool.tool_rule for tool in self._tools]
