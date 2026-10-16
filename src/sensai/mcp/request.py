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


def add_tool_result(payload: dict[str, Any], tool_call: dict, tool_result: dict | None) -> dict[str, Any]:
    """Return a follow-up request containing one model tool call and its result."""
    messages = list(payload.get("messages", []))
    messages.append({"role": "assistant", "tool_calls": [tool_call]})
    if tool_result is not None:
        messages.append({"role": "tool", "content": str(tool_result.get("content", ""))})

    follow_up = dict(payload)
    follow_up["messages"] = messages
    return follow_up
