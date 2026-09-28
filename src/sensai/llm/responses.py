class ShowResponse:
    """Response of the show call format as adapter."""

    model: str
    tools: bool
    think: bool

    def __init__(self, model: str, *, tools: bool = False, think: bool = False) -> None:
        """Init the ShowResponse class."""
        self.model = model
        self.tools = tools
        self.think = think


class ChatResponse:
    """Response of the Chat to format as adapter."""

    model: str = ""
    done: bool = False
    role: str = ""
    content: str | None = None
    thinking: str | None = None
    tool_calls: list[dict] | None = None

    def __init__(self, payload: dict) -> None:
        """Init from /api/chat response, streamed or not."""
        self.model = payload.get("model", "")
        self.done = payload.get("done", True)
        message: dict = payload.get("message", {})
        if message == {}:
            return
        self.role = message.get("role", "")
        self.content = message.get("content", "")
        self.thinking = message.get("thinking", "")
        self.tool_calls = message.get("tool_calls", [])
