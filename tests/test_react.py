from sensai.llm.responses import ChatResponse, ShowResponse
from sensai.llm.thinking.react import ReAct
from sensai.monitoring.interaction import Interaction, InteractionStatus
from sensai.tools.ask_user import AskUserTool
from sensai.tools.tool import Tool
from sensai.tools.tool_manager import ToolManager
from sensai.ui.adapter import Confirmation


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


class ToolCallProvider(FakeProvider):
    def __init__(self, name: str, arguments: dict | None = None) -> None:
        super().__init__()
        self._call = {"function": {"name": name, "arguments": arguments or {}}}

    def chat(self, payload: dict, *, stream: bool) -> list[ChatResponse]:
        if "tools" not in payload:
            return super().chat(payload, stream=stream)
        self.payloads.append(payload)
        message = {"role": "assistant", "content": "", "tool_calls": [self._call]}
        return [ChatResponse({"done": True, "message": message})]


class ThoughtProvider(FakeProvider):
    """Thinks `thought` on the first request, then answers like `FakeProvider`."""

    def __init__(self, thought: str) -> None:
        super().__init__()
        self._thought = thought

    def chat(self, payload: dict, *, stream: bool) -> list[ChatResponse]:
        if self.payloads:
            return super().chat(payload, stream=stream)
        self.payloads.append(payload)
        return [ChatResponse({"done": True, "message": {"role": "assistant", "content": self._thought}})]


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


class TestConfirmation:
    def run_gated(self, answer: Confirmation) -> tuple[ToolCallProvider, str]:
        provider = ToolCallProvider("search")
        tools = ToolManager([SearchTool()], confirm=("search",), approver=lambda _name, _args: answer)
        react = ReAct(FakeProviderManager(provider), tools, 1)  # type: ignore[arg-type]
        chunks = list(react.generate_answer([{"role": "user", "content": "Find it"}]))
        return provider, "".join(chunk.content or "" for chunk in chunks)

    def test_refusal_is_sent_to_the_model_and_the_loop_still_answers(self) -> None:
        provider, answer = self.run_gated(Confirmation.DENY)

        assert {
            "role": "tool",
            "tool_name": "search",
            "content": "error: the user declined to run 'search'; do not call it again, answer without it",
        } in provider.payloads[-1]["messages"]
        assert answer == "Bonjour !"

    def test_allowed_call_sends_the_tool_result_to_the_model(self) -> None:
        provider, _ = self.run_gated(Confirmation.ALLOW)

        assert {"role": "tool", "tool_name": "search", "content": "a result"} in provider.payloads[-1]["messages"]


class TestAskUser:
    def test_answer_of_the_user_reaches_the_final_request(self) -> None:
        provider = ToolCallProvider("ask_user", {"question": "Which quarter?"})
        tools = ToolManager([AskUserTool(lambda _question: "Q2 2026")])
        react = ReAct(FakeProviderManager(provider), tools, 1)  # type: ignore[arg-type]

        list(react.generate_answer([{"role": "user", "content": "How did we do?"}]))

        assert {"role": "tool", "tool_name": "ask_user", "content": "Q2 2026"} in provider.payloads[-1]["messages"]

    def test_is_listed_in_the_first_think_instruction(self) -> None:
        provider = FakeProvider()
        tools = ToolManager([AskUserTool(lambda _question: None)])
        react = ReAct(FakeProviderManager(provider), tools, 1)  # type: ignore[arg-type]

        list(react.generate_answer([{"role": "user", "content": "How did we do?"}]))

        assert "- ask_user: Ask the user one short question" in provider.payloads[0]["messages"][-1]["content"]


def tools_offered_after(thought: str) -> bool:
    provider = ThoughtProvider(thought)
    react = ReAct(FakeProviderManager(provider), ToolManager([SearchTool()]), 1)  # type: ignore[arg-type]
    list(react.generate_answer([{"role": "user", "content": "Bonjour"}]))
    return "tools" in provider.payloads[1]


class TestToolDecision:
    def test_yes_offers_the_tools(self) -> None:
        assert tools_offered_after("I need a recent fact.\nTool needed: yes")

    def test_no_does_not_offer_the_tools(self) -> None:
        assert not tools_offered_after("It is a greeting.\nTool needed: no")

    def test_no_decision_offers_the_tools(self) -> None:
        assert tools_offered_after("I am not sure.")

    def test_quoting_the_instruction_does_not_count_as_a_refusal(self) -> None:
        thought = "I must end with 'Tool needed: yes' or 'Tool needed: no'. I lack the data.\nTool needed: yes"

        assert tools_offered_after(thought)

    def test_only_the_last_decision_counts(self) -> None:
        assert not tools_offered_after("Tool needed: yes? No, it is arithmetic.\nTool needed: no")

    def test_reads_a_decision_in_markdown(self) -> None:
        assert not tools_offered_after("It is a greeting.\n**Tool needed:** No")


class TestAnswerDecision:
    def test_searches_again_only_when_the_last_decision_is_no(self) -> None:
        thought = "End with 'Answer found: yes' or 'Answer found: no'. The page has the figure.\nAnswer found: yes"

        assert not ReAct._wants_tool(thought, after_observation=True)  # noqa: SLF001
        assert ReAct._wants_tool("The page is empty.\nAnswer found: no", after_observation=True)  # noqa: SLF001

    def test_answers_without_a_decision(self) -> None:
        assert not ReAct._wants_tool("The page has the figure.", after_observation=True)  # noqa: SLF001


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
