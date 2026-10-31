from dataclasses import dataclass

_NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class ChatMetrics:
    """Token counts and durations reported by the provider on the last chunk of a chat call."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_ms: float | None = None
    load_duration_ms: float | None = None
    prompt_eval_duration_ms: float | None = None
    eval_duration_ms: float | None = None

    @classmethod
    def from_payload(cls, payload: dict) -> ChatMetrics:
        """Build from an Ollama /api/chat payload, whose durations are in nanoseconds."""
        return cls(
            prompt_tokens=payload.get("prompt_eval_count"),
            completion_tokens=payload.get("eval_count"),
            total_duration_ms=_ns_to_ms(payload.get("total_duration")),
            load_duration_ms=_ns_to_ms(payload.get("load_duration")),
            prompt_eval_duration_ms=_ns_to_ms(payload.get("prompt_eval_duration")),
            eval_duration_ms=_ns_to_ms(payload.get("eval_duration")),
        )


def _ns_to_ms(value: int | None) -> float | None:
    return None if value is None else value / _NS_PER_MS


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
    metrics: ChatMetrics | None = None

    error: str | None = None

    def __init__(self, payload: dict) -> None:
        """Init from /api/chat response, streamed or not."""
        self.done = payload.get("done", True)
        self.error = payload.get("error")
        if self.error is not None:
            return

        self.model = payload.get("model", "")
        if self.done:
            self.metrics = ChatMetrics.from_payload(payload)

        message: dict = payload.get("message", {})
        if message == {}:
            return
        self.role = message.get("role", "")
        self.content = message.get("content") or None
        self.thinking = message.get("thinking") or None
        self.tool_calls = message.get("tool_calls", [])
