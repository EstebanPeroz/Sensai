from collections.abc import Callable

HandlerFunc = Callable[[str], str | None]


class ChatPipeline:
    """Run a message through an ordered chain of handlers."""

    def __init__(self) -> None:
        """Start with an empty chain."""
        self._handlers: list[HandlerFunc] = []

    def use(self, handler: HandlerFunc) -> None:
        """Append a new handler to the end of the chain."""
        self._handlers.append(handler)

    def execute(self, message: str) -> str:
        """Run the message through the pipeline until a handler returns a response."""
        for handler in self._handlers:
            response = handler(message)
            if response is not None:
                return response
        return "Bot: I'm sorry, I didn't quite catch that."
