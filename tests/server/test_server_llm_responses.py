import pytest
from sensai_server.llm.responses import ChatMetrics, ChatResponse, ShowResponse


class TestChatMetrics:
    def test_converts_durations_from_nanoseconds(self) -> None:
        payload = {
            "prompt_eval_count": 12,
            "eval_count": 3,
            "total_duration": 2_000_000,
            "load_duration": 500_000,
            "prompt_eval_duration": 250_000,
            "eval_duration": 1_500_000,
        }

        assert ChatMetrics.from_payload(payload) == ChatMetrics(
            prompt_tokens=12,
            completion_tokens=3,
            total_duration_ms=2.0,
            load_duration_ms=0.5,
            prompt_eval_duration_ms=0.25,
            eval_duration_ms=1.5,
        )

    def test_missing_fields_stay_none(self) -> None:
        assert ChatMetrics.from_payload({}) == ChatMetrics()

    def test_zero_duration_is_kept(self) -> None:
        assert ChatMetrics.from_payload({"load_duration": 0}).load_duration_ms == 0.0


class TestChatResponse:
    def test_reads_the_message(self) -> None:
        response = ChatResponse(
            {
                "model": "qwen3:1.7b",
                "done": False,
                "message": {"role": "assistant", "content": "hi", "thinking": "hm", "tool_calls": [{"x": 1}]},
            }
        )

        assert (response.model, response.done, response.role) == ("qwen3:1.7b", False, "assistant")
        assert (response.content, response.thinking, response.tool_calls) == ("hi", "hm", [{"x": 1}])
        assert response.metrics is None
        assert response.error is None

    def test_empty_texts_become_none(self) -> None:
        response = ChatResponse({"done": False, "message": {"role": "assistant", "content": "", "thinking": ""}})

        assert response.content is None
        assert response.thinking is None
        assert response.tool_calls == []

    def test_done_defaults_to_true_and_carries_metrics(self) -> None:
        response = ChatResponse({"model": "m", "eval_count": 4})

        assert response.done is True
        assert response.metrics == ChatMetrics(completion_tokens=4)

    def test_without_message_keeps_the_defaults(self) -> None:
        response = ChatResponse({"model": "m", "done": True})

        assert (response.role, response.content, response.thinking, response.tool_calls) == ("", None, None, None)

    def test_error_skips_everything_else(self) -> None:
        response = ChatResponse({"error": "model not found", "model": "m", "message": {"content": "x"}})

        assert response.error == "model not found"
        assert response.done is True
        assert (response.model, response.content, response.metrics) == ("", None, None)

    def test_instances_do_not_share_state(self) -> None:
        first = ChatResponse({"done": False, "message": {"content": "a"}})
        second = ChatResponse({"done": False})

        assert first.content == "a"
        assert second.content is None


class TestShowResponse:
    def test_defaults_to_no_capability(self) -> None:
        response = ShowResponse("m")

        assert (response.model, response.tools, response.think) == ("m", False, False)

    @pytest.mark.parametrize(("tools", "think"), [(True, False), (False, True), (True, True)])
    def test_keeps_capabilities(self, *, tools: bool, think: bool) -> None:
        response = ShowResponse("m", tools=tools, think=think)

        assert (response.tools, response.think) == (tools, think)
