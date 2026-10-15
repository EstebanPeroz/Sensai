import json
import logging
from typing import TYPE_CHECKING, Any

from sensai.error import SensaiError

if TYPE_CHECKING:
    from sensai.tools.tool import Tool

logger = logging.getLogger(__name__)


class ToolManager:
    """Tmp."""

    _tools: list[Tool]

    def __init__(self, tools: list[Tool] | None = None) -> None:
        """Tmp."""
        self._tools: list[Tool] = tools or []

    def call(self, tool_call: dict[str, Any]) -> dict[str, str]:
        """Run the tool named in an LLM-style function-call payload and return its result as a tool message."""
        function = tool_call.get("function") or {}
        name = function.get("name")
        if name is None:
            return self._message("unknown", "error: malformed tool call, missing function name")
        tool = next((tool for tool in self._tools if tool.name == name), None)
        if tool is None:
            return self._message(name, f"error: unknown tool '{name}'")
        try:
            result = tool.run(**(function.get("arguments") or {}))
        except (SensaiError, TypeError) as err:
            return self._message(name, f"error: {err}")
        except Exception as err:
            logger.exception("Tool '%s' failed unexpectedly", name)
            return self._message(name, f"error: {type(err).__name__}: {err}")
        return self._message(name, result if isinstance(result, str) else json.dumps(result, default=str))

    def get_tool_call_rules(self) -> list[dict]:
        """Tmp."""
        return [tool.tool_rule for tool in self._tools]

    @staticmethod
    def _message(tool_name: str, content: str) -> dict[str, str]:
        return {"role": "tool", "tool_name": tool_name, "content": content}
