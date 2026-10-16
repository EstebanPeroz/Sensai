from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sensai.tools.tool_manager import ToolManager


def add_tool_schemas(payload: dict[str, Any], tool_manager: ToolManager) -> None:
    """Add enabled tool schemas to an LLM request when tools are available."""
    if rules := tool_manager.get_tool_call_rules():
        payload["tools"] = rules
