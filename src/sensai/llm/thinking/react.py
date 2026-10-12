from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sensai.llm.provider_manager import ProviderManager
    from sensai.llm.responses import ChatResponse
    from sensai.tools.tool_manager import ToolManager


class ReAct:
    """Implement thinking method "ReAct" for reasoning and acting in a loop."""

    def __init__(self, provider_manager: ProviderManager, tool_manager: ToolManager, max_iterations: int = 5) -> None:
        """Bind the ReAct loop to a `provider_manager` and a `tool_manager`, with a maximum of `max_iterations`."""
        self._provider_manager = provider_manager
        self._tool_manager = tool_manager
        self._max_iterations = max_iterations

    def generate_answer(self, messages: list[dict]) -> Iterator[ChatResponse]:
        """Stream the answer to `messages`, running the tools the model asks for between two requests.

        Builds the answer using the ReAct loop.
        """
        provider = self._provider_manager.provider()
        model = provider.current_model()
        capabilities = provider.show(model)
        payload = {"model": model, "messages": messages}
        rules = self._tool_manager.get_tool_call_rules() if capabilities.tools else []

        # Only enables our thinking loop
        payload["think"] = False
        for _ in range(self._max_iterations if rules else 0):
            content = ""
            tool_calls = []
            payload_with_tools = payload.copy()
            payload_with_tools["tools"] = rules
            for chunk in provider.chat(payload_with_tools, stream=True):
                yield chunk
                content += chunk.content or ""
                tool_calls.extend(chunk.tool_calls or [])
            if not tool_calls:
                return
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            messages.extend(self._tool_manager.call(call) for call in tool_calls)

        for chunk in provider.chat(payload, stream=True):
            yield chunk
