from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sensai.tools.tool_manager import ToolManager


def add_tool_schemas(payload: dict[str, Any], tool_manager: ToolManager) -> None:
    """Add enabled tool schemas to an LLM request when tools are available."""
    if rules := tool_manager.get_tool_call_rules():
        payload["tools"] = rules


def execute_one_tool_call(tool_calls: list[dict], tool_manager: ToolManager) -> dict | None:
    """Execute the first model-requested tool call, without starting another model round."""
    if not tool_calls:
        return None
    return tool_manager.call(tool_calls[0])
