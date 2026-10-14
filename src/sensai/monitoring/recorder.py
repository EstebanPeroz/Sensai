import time
from typing import TYPE_CHECKING

from sensai.llm.responses import ChatMetrics
from sensai.monitoring.interaction import Interaction, InteractionStatus

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from sensai.llm.responses import ChatResponse

_MS_PER_S = 1000


class InteractionRecorder:
    """Accumulates the chunks of one streamed call to the model into an `Interaction`.

    The clock starts when the recorder is created, so create it right before sending the request.
    """

    def __init__(
        self,
        *,
        model: str,
        request: list[dict],
        branch_id: UUID | None = None,
        persona: str | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Start timing a call sending `request` to `model`; `clock` returns seconds."""
        self._model = model
        self._request = request
        self._branch_id = branch_id
        self._persona = persona
        self._clock = clock
        self._start = clock()
        self._first_token_ms: float | None = None
        self._content = ""
        self._thinking = ""
        self._tool_calls: list[dict] = []
        self._metrics = ChatMetrics()

    @property
    def content(self) -> str:
        """Response text received so far."""
        return self._content

    def add_chunk(self, chunk: ChatResponse) -> None:
        """Accumulate the content, thinking, tool calls and metrics carried by `chunk`."""
        if self._first_token_ms is None and (chunk.content or chunk.thinking or chunk.tool_calls):
            self._first_token_ms = self._elapsed_ms()
        if chunk.content is not None:
            self._content += chunk.content
        if chunk.thinking is not None:
            self._thinking += chunk.thinking
        if chunk.tool_calls:
            self._tool_calls.extend(chunk.tool_calls)
        if chunk.metrics is not None:
            self._metrics = chunk.metrics

    def finish(self, status: InteractionStatus, error: str | None = None) -> Interaction:
        """Stop the clock and return the record of the call, ended with `status`."""
        return Interaction(
            model=self._model,
            status=status,
            latency_ms=self._elapsed_ms(),
            branch_id=self._branch_id,
            persona=self._persona,
            request=self._request,
            response=self._content or None,
            thinking=self._thinking or None,
            tool_calls=self._tool_calls,
            error=error,
            first_token_ms=self._first_token_ms,
            metrics=self._metrics,
        )

    def _elapsed_ms(self) -> float:
        return (self._clock() - self._start) * _MS_PER_S
