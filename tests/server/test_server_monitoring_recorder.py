import pytest
from sensai_server.llm.responses import ChatResponse
from sensai_server.monitoring.interaction import InteractionStatus
from sensai_server.monitoring.recorder import InteractionRecorder

_REQUEST = [{"role": "user", "content": "hi"}]
_CALL = {"function": {"name": "web_search", "arguments": {"query": "x"}}}


def chunk(content: str = "", *, thinking: str = "", tool_calls: list | None = None) -> ChatResponse:
    message = {"role": "assistant", "content": content, "thinking": thinking, "tool_calls": tool_calls or []}
    return ChatResponse({"model": "m", "message": message, "done": False})


class Clock:
    """A fake clock, in seconds, that the test moves forward."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class TestInteractionRecorder:
    def test_records_a_streamed_call(self) -> None:
        time = Clock(10.0)
        rec = InteractionRecorder(model="m", request=_REQUEST, clock=time)
        time.now = 10.1
        rec.add_chunk(chunk(""))  # empty: not the first token
        time.now = 10.2
        rec.add_chunk(chunk("Hel", thinking="hmm"))
        rec.add_chunk(chunk("lo", tool_calls=[_CALL]))
        rec.add_chunk(ChatResponse({"model": "m", "message": {}, "done": True, "eval_count": 2}))
        time.now = 10.5

        interaction = rec.finish(InteractionStatus.OK)

        assert (interaction.model, interaction.status, interaction.error) == ("m", InteractionStatus.OK, None)
        assert interaction.request == _REQUEST
        assert (interaction.response, interaction.thinking, interaction.tool_calls) == ("Hello", "hmm", [_CALL])
        assert interaction.first_token_ms == pytest.approx(200.0)
        assert interaction.latency_ms == pytest.approx(500.0)
        assert interaction.metrics.completion_tokens == 2

    def test_records_a_call_failing_before_any_chunk(self) -> None:
        time = Clock()
        rec = InteractionRecorder(model="m", request=_REQUEST, clock=time)
        time.now = 0.05

        interaction = rec.finish(InteractionStatus.ERROR, "connection refused")

        assert (interaction.status, interaction.error) == (InteractionStatus.ERROR, "connection refused")
        assert (interaction.response, interaction.thinking, interaction.first_token_ms) == (None, None, None)
        assert interaction.latency_ms == pytest.approx(50.0)

    def test_keeps_the_last_metrics(self) -> None:
        rec = InteractionRecorder(model="m", request=_REQUEST, clock=Clock())
        rec.add_chunk(ChatResponse({"model": "m", "message": {}, "done": True, "eval_count": 1}))
        rec.add_chunk(ChatResponse({"model": "m", "message": {}, "done": True, "eval_count": 3}))
        rec.add_chunk(chunk("late"))  # no metrics: the last ones are kept

        assert rec.finish(InteractionStatus.OK).metrics.completion_tokens == 3

    def test_stores_empty_content_as_none(self) -> None:
        rec = InteractionRecorder(model="m", request=_REQUEST, clock=Clock())
        rec.add_chunk(chunk(""))
        rec.add_chunk(chunk("", thinking=""))

        interaction = rec.finish(InteractionStatus.OK)

        assert (interaction.response, interaction.thinking, interaction.tool_calls) == (None, None, [])
        assert interaction.first_token_ms is None
