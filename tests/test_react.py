from sensai.llm.responses import ChatResponse, ShowResponse
from sensai.llm.thinking.react import ReAct
from sensai.monitoring.interaction import Interaction, InteractionStatus
from sensai.tools.tool import Tool
from sensai.tools.tool_manager import ToolManager


class SearchTool(Tool[str]):
    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search the web."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def run(self, **kwargs: object) -> str:
        del kwargs
        return "a result"


class FakeProvider:
    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self.shown: list[str] = []

    def current_model(self) -> str:
        return "fake-model"

    def show(self, model_name: str) -> ShowResponse:
        self.shown.append(model_name)
        return ShowResponse(model_name, tools=True)

    def chat(self, payload: dict, *, stream: bool) -> list[ChatResponse]:
        assert stream
        self.payloads.append(payload)
        return [ChatResponse({"done": True, "message": {"role": "assistant", "content": "Bonjour !"}})]


class MalformedCallProvider(FakeProvider):
    def chat(self, payload: dict, *, stream: bool) -> list[ChatResponse]:
        if "tools" not in payload:
            return super().chat(payload, stream=stream)
        self.payloads.append(payload)
        message = {"role": "assistant", "content": "", "tool_calls": [{"function": "search"}]}
        return [ChatResponse({"done": True, "message": message})]


class FakeProviderManager:
    def __init__(self, provider: FakeProvider) -> None:
        self._provider = provider

    def provider(self) -> FakeProvider:
        return self._provider


def run(max_iterations: int, tools: list[Tool] | None = None) -> tuple[FakeProvider, str]:
    provider = FakeProvider()
    react = ReAct(FakeProviderManager(provider), ToolManager(tools), max_iterations)  # type: ignore[arg-type]
    chunks = list(react.generate_answer([{"role": "user", "content": "Bonjour"}]))
    return provider, "".join(chunk.content or "" for chunk in chunks)


class TestEffortOff:
    def test_answers_with_a_single_request_without_tools(self) -> None:
        provider, answer = run(max_iterations=0)

        assert answer == "Bonjour !"
        assert len(provider.payloads) == 1
        assert "tools" not in provider.payloads[0]

    def test_does_not_look_up_the_model_capabilities(self) -> None:
        provider, _ = run(max_iterations=0)

        assert provider.shown == []

    def test_capabilities_are_looked_up_when_thinking_is_allowed(self) -> None:
        provider, _ = run(max_iterations=1, tools=[SearchTool()])

        assert provider.shown == ["fake-model"]


class TestWithoutTools:
    def test_does_not_look_up_the_model_capabilities(self) -> None:
        provider, _ = run(max_iterations=1)

        assert provider.shown == []

    def test_still_answers(self) -> None:
        provider, answer = run(max_iterations=1)

        assert answer == "Bonjour !"
        assert all("tools" not in payload for payload in provider.payloads)


class TestMalformedToolCall:
    def test_is_reported_to_the_model_instead_of_crashing(self) -> None:
        provider = MalformedCallProvider()
        react = ReAct(FakeProviderManager(provider), ToolManager([SearchTool()]), 1)  # type: ignore[arg-type]

        chunks = list(react.generate_answer([{"role": "user", "content": "Find it"}]))

        sent = provider.payloads[-1]["messages"]
        assert {
            "role": "tool",
            "tool_name": "unknown",
            "content": "error: malformed tool call, function must be an object",
        } in sent
        assert "-> Using unknown...\n" in [chunk.thinking for chunk in chunks]


class TestInteractions:
    def test_records_every_call_with_its_request(self) -> None:
        provider = MalformedCallProvider()
        react = ReAct(FakeProviderManager(provider), ToolManager([SearchTool()]), 1)  # type: ignore[arg-type]
        interactions: list[Interaction] = []

        list(react.generate_answer([{"role": "user", "content": "Find it"}], interactions.append))

        assert [interaction.request for interaction in interactions] == [
            payload["messages"] for payload in provider.payloads
        ]
        assert [interaction.status for interaction in interactions] == [InteractionStatus.OK] * len(provider.payloads)
        assert interactions[1].tool_calls == [{"function": "search"}]

    def test_records_a_call_closed_before_its_end_as_interrupted(self) -> None:
        react = ReAct(FakeProviderManager(FakeProvider()), ToolManager(), 1)  # type: ignore[arg-type]
        interactions: list[Interaction] = []

        chunks = react.generate_answer([{"role": "user", "content": "Bonjour"}], interactions.append)
        next(chunks)
        chunks.close()

        assert [interaction.status for interaction in interactions] == [InteractionStatus.INTERRUPTED]


class TestContextWindow:
    def test_is_asked_of_the_model_on_every_call(self) -> None:
        provider = FakeProvider()
        react = ReAct(FakeProviderManager(provider), ToolManager(), 1, context_window=8192)  # type: ignore[arg-type]

        list(react.generate_answer([{"role": "user", "content": "Bonjour"}]))

        assert [payload["options"] for payload in provider.payloads] == [{"num_ctx": 8192}] * len(provider.payloads)

    def test_is_left_to_the_provider_without_one(self) -> None:
        provider, _ = run(max_iterations=1)

        assert all("options" not in payload for payload in provider.payloads)
