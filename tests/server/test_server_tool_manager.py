import asyncio
from typing import Any

import pytest
from sensai_protocol import ServerMessage
from sensai_protocol.client_messages import Confirmation
from sensai_server.tools.state import ToolState
from sensai_server.tools.tool import Tool, ToolContext, get_parameter
from sensai_server.tools.tool_manager import ToolManager

from tests.server.fakes import RecordingOutbound


class ScriptedOutbound(RecordingOutbound):
    """Answers each confirmation with the next scripted answer, recording what was asked and any overlap."""

    def __init__(self, *answers: Confirmation, delay: float = 0) -> None:
        super().__init__()
        self._answers = list(answers)
        self._delay = delay
        self._asking = 0
        self.requests: list[tuple[str, dict]] = []
        self.overlapped = False

    async def ask_confirmation(self, tool_name: str, arguments: dict[str, Any]) -> Confirmation:
        self.requests.append((tool_name, arguments))
        self._asking += 1
        self.overlapped |= self._asking > 1
        await asyncio.sleep(self._delay)
        self._asking -= 1
        return self._answers.pop(0)


class ClosedOutbound(RecordingOutbound):
    """An outbound whose client is gone: nothing can be sent."""

    async def send(self, message: ServerMessage) -> bool:
        del message
        return False


class EchoTool(Tool[str]):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Return the given city."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}

    async def run(self, ctx: ToolContext, **kwargs: object) -> str:
        del ctx
        return get_parameter("city", str, **kwargs)


class AddTool(Tool[int]):
    @property
    def name(self) -> str:
        return "add"

    @property
    def description(self) -> str:
        return "Add two integers."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }

    async def run(self, ctx: ToolContext, a: int, b: int) -> int:  # type: ignore[override]
        del ctx
        return a + b


class ResultTool(Tool[object]):
    def __init__(self, result: object, name: str = "result") -> None:
        self._result = result
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Return a fixed result."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def run(self, ctx: ToolContext, **kwargs: object) -> object:
        del ctx, kwargs
        return self._result


class NamedTool(ResultTool):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} ran", name)


class GatedTool(NamedTool):
    @property
    def requires_confirmation(self) -> bool:
        return True


class BrokenTool(NamedTool):
    def __init__(self) -> None:
        super().__init__("broken")

    async def run(self, ctx: ToolContext, **kwargs: object) -> str:
        del ctx, kwargs
        msg = "page could not be parsed"
        raise ValueError(msg)


class CancelledTool(NamedTool):
    def __init__(self) -> None:
        super().__init__("cancelled")

    async def run(self, ctx: ToolContext, **kwargs: object) -> str:
        del ctx, kwargs
        raise asyncio.CancelledError


class HandshakeTool(NamedTool):
    """Sets its own event, then waits for the other's: two of them only finish when they run in parallel."""

    def __init__(self, name: str, mine: asyncio.Event, other: asyncio.Event) -> None:
        super().__init__(name)
        self._mine = mine
        self._other = other

    async def run(self, ctx: ToolContext, **kwargs: object) -> str:
        del ctx, kwargs
        self._mine.set()
        await self._other.wait()
        return f"{self.name} ran"


class BlockingTool(NamedTool):
    """Waits forever once started, recording whether it got cancelled."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.started = asyncio.Event()
        self.cancelled = False

    async def run(self, ctx: ToolContext, **kwargs: object) -> str:
        del ctx, kwargs
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return "never"


def make_call(name: str, arguments: dict | None = None) -> dict:
    return {"type": "function", "function": {"index": 0, "name": name, "arguments": arguments or {}}}


def message(name: str, content: str) -> dict[str, str]:
    return {"role": "tool", "tool_name": name, "content": content}


def declined(name: str) -> dict[str, str]:
    return message(name, f"error: the user declined to run '{name}'; do not call it again, answer without it")


async def call(manager: ToolManager, tool_call: object, outbound: RecordingOutbound | None = None) -> dict[str, str]:
    ctx = ToolContext(out=outbound or RecordingOutbound())
    return await manager.call(tool_call, manager.new_state(), ctx)  # type: ignore[arg-type]


def make_manager() -> ToolManager:
    return ToolManager([EchoTool(), AddTool(), ResultTool({"city": "Paris", "age": 30}, "profile")])


def make_pair(disabled: tuple[str, ...] = ()) -> ToolManager:
    return ToolManager([NamedTool("weather"), NamedTool("echo")], disabled=disabled)


def enabled_names(manager: ToolManager, state: ToolState) -> list[str]:
    return [name for name in manager.names() if manager.is_enabled(state, name)]


def rule_names(manager: ToolManager, state: ToolState) -> list[str]:
    return [rule["function"]["name"] for rule in manager.get_tool_call_rules(state)]


class TestCall:
    async def test_calls_matching_tool_with_arguments(self) -> None:
        assert await call(make_manager(), make_call("echo", {"city": "Paris"})) == message("echo", "Paris")

    async def test_unknown_tool_returns_error_message(self) -> None:
        result = await call(make_manager(), make_call("ghost"))

        assert result == message("ghost", "error: unknown tool 'ghost'")

    @pytest.mark.parametrize(
        "tool_call",
        ["web_search", {"type": "function"}, {"type": "function", "function": {"arguments": {}}}],
    )
    async def test_call_without_function_name_returns_error_message(self, tool_call: object) -> None:
        result = await call(make_manager(), tool_call)

        assert result == message("unknown", "error: malformed tool call, missing function name")

    @pytest.mark.parametrize("function", ["web_search", ["web_search"], 3])
    async def test_function_that_is_not_an_object_returns_error_message(self, function: object) -> None:
        result = await call(make_manager(), {"type": "function", "function": function})

        assert result == message("unknown", "error: malformed tool call, function must be an object")

    async def test_wrong_argument_type_returns_error_message(self) -> None:
        result = await call(make_manager(), make_call("echo", {"city": 2}))

        assert result == message("echo", "error: Parameter 'city' must be str, got int")

    async def test_missing_arguments_are_treated_as_empty(self) -> None:
        result = await call(make_manager(), {"type": "function", "function": {"name": "echo", "arguments": None}})

        assert result == message("echo", "error: Miss parameter: city")

    async def test_unexpected_argument_returns_error_message(self) -> None:
        result = await call(make_manager(), make_call("add", {"a": 1, "b": 2, "c": 3}))

        assert result["content"].startswith("error: ")
        assert "unexpected keyword argument 'c'" in result["content"]

    async def test_number_result_is_converted_to_text(self) -> None:
        assert await call(make_manager(), make_call("add", {"a": 1, "b": 2})) == message("add", "3")

    async def test_dict_result_is_converted_to_json_text(self) -> None:
        result = await call(make_manager(), make_call("profile"))

        assert result == message("profile", '{"city": "Paris", "age": 30}')


class TestResultConversion:
    async def test_result_that_cannot_become_json_returns_error_message(self) -> None:
        result = await call(ToolManager([ResultTool({("a", "b"): 1})]), make_call("result"))

        assert result == message("result", "error: keys must be str, int, float, bool or None, not tuple")

    async def test_circular_result_returns_error_message(self) -> None:
        circular: list[object] = []
        circular.append(circular)

        result = await call(ToolManager([ResultTool(circular)]), make_call("result"))

        assert result == message("result", "error: ValueError: Circular reference detected")

    async def test_unserializable_values_are_converted_to_text(self) -> None:
        result = await call(ToolManager([ResultTool({"when": object})]), make_call("result"))

        assert result["content"] == '{"when": "<class \'object\'>"}'


class TestToolName:
    def test_reads_the_name_of_a_well_formed_call(self) -> None:
        assert ToolManager.tool_name(make_call("echo")) == "echo"

    @pytest.mark.parametrize(
        "tool_call",
        [
            "web_search",
            {"type": "function"},
            {"type": "function", "function": "web_search"},
            {"type": "function", "function": {"arguments": {}}},
        ],
    )
    def test_is_unknown_for_a_malformed_call(self, tool_call: object) -> None:
        assert ToolManager.tool_name(tool_call) == "unknown"


class TestUnexpectedFailure:
    async def test_unexpected_exception_returns_error_message(self) -> None:
        result = await call(ToolManager([BrokenTool()]), make_call("broken"))

        assert result == message("broken", "error: ValueError: page could not be parsed")

    async def test_unexpected_exception_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        await call(ToolManager([BrokenTool()]), make_call("broken"))

        assert "Tool 'broken' failed unexpectedly" in caplog.text

    async def test_cancellation_is_not_swallowed(self) -> None:
        with pytest.raises(asyncio.CancelledError):
            await call(ToolManager([CancelledTool()]), make_call("cancelled"))


class TestEnableAndDisable:
    def test_disable_then_enable_restores_the_tool(self) -> None:
        manager = make_pair()
        state = manager.new_state()

        assert manager.disable(state, "echo") is True
        assert manager.is_enabled(state, "echo") is False
        assert manager.enable(state, "echo") is True
        assert manager.is_enabled(state, "echo") is True

    def test_unknown_name_is_not_enabled(self) -> None:
        manager = make_pair()

        assert manager.is_enabled(manager.new_state(), "ghost") is False

    def test_enable_and_disable_are_idempotent(self) -> None:
        manager = make_pair()
        state = manager.new_state()

        manager.disable(state, "echo")
        manager.disable(state, "echo")
        assert enabled_names(manager, state) == ["weather"]
        manager.enable(state, "echo")
        manager.enable(state, "echo")
        assert enabled_names(manager, state) == ["echo", "weather"]

    def test_unknown_names_are_rejected(self) -> None:
        manager = make_pair()
        state = manager.new_state()

        assert manager.enable(state, "ghost") is False
        assert manager.disable(state, "ghost") is False
        assert state.enabled == {"echo", "weather"}

    def test_a_name_given_as_disabled_but_unknown_stays_unknown(self) -> None:
        manager = make_pair(disabled=("ghost",))

        assert manager.enable(manager.new_state(), "ghost") is False

    def test_toggling_in_one_state_leaves_the_others_untouched(self) -> None:
        manager = make_pair()
        first, second = manager.new_state(), manager.new_state()

        manager.disable(first, "echo")

        assert enabled_names(manager, first) == ["weather"]
        assert enabled_names(manager, second) == ["echo", "weather"]


class TestGetToolCallRules:
    def test_returns_the_rules_of_the_enabled_tools_sorted(self) -> None:
        manager = ToolManager([NamedTool("zebra"), NamedTool("alpha")])

        assert rule_names(manager, manager.new_state()) == ["alpha", "zebra"]

    def test_rules_follow_a_tool_being_disabled(self) -> None:
        manager = make_pair()
        state = manager.new_state()

        manager.disable(state, "echo")

        assert rule_names(manager, state) == ["weather"]

    def test_no_rule_is_returned_when_every_tool_is_disabled(self) -> None:
        manager = make_pair(disabled=("echo", "weather"))

        assert manager.get_tool_call_rules(manager.new_state()) == []

    def test_rules_follow_the_chat_endpoint_format(self) -> None:
        manager = ToolManager([EchoTool()])

        assert manager.get_tool_call_rules(manager.new_state()) == [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Return the given city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            },
        ]


class TestCallOnDisabledTool:
    async def test_disabled_tool_is_unreachable(self) -> None:
        # A missing `city` would raise if the tool ran; being disabled must short-circuit first.
        result = await call(ToolManager([EchoTool()], disabled=("echo",)), make_call("echo"))

        assert result == message("echo", "error: unknown tool 'echo'")

    async def test_tool_disabled_in_the_state_is_unreachable(self) -> None:
        manager = make_manager()
        state = manager.new_state()
        manager.disable(state, "echo")

        result = await manager.call(make_call("echo", {"city": "Paris"}), state, ToolContext(out=RecordingOutbound()))

        assert result == message("echo", "error: unknown tool 'echo'")

    async def test_manager_without_tools_reaches_nothing(self) -> None:
        assert await call(ToolManager(), make_call("echo")) == message("echo", "error: unknown tool 'echo'")


class TestConfirmation:
    async def test_allowed_call_runs_the_tool(self) -> None:
        outbound = ScriptedOutbound(Confirmation.ALLOW)
        manager = ToolManager([EchoTool()], confirm=("echo",))

        result = await call(manager, make_call("echo", {"city": "Paris"}), outbound)

        assert result == message("echo", "Paris")
        assert outbound.requests == [("echo", {"city": "Paris"})]

    async def test_denied_call_does_not_run_the_tool(self) -> None:
        # A missing `city` would raise if the tool ran; the refusal must short-circuit first.
        result = await call(
            ToolManager([EchoTool()], confirm=("echo",)), make_call("echo"), ScriptedOutbound(Confirmation.DENY)
        )

        assert result == declined("echo")

    async def test_allow_asks_again_on_the_next_call(self) -> None:
        manager = ToolManager([NamedTool("weather")], confirm=("weather",))
        state = manager.new_state()
        outbound = ScriptedOutbound(Confirmation.ALLOW, Confirmation.DENY)
        ctx = ToolContext(out=outbound)

        await manager.call(make_call("weather"), state, ctx)

        assert await manager.call(make_call("weather"), state, ctx) == declined("weather")
        assert len(outbound.requests) == 2

    async def test_always_asks_only_once_in_its_session(self) -> None:
        manager = ToolManager([NamedTool("weather")], confirm=("weather",))
        state = manager.new_state()
        outbound = ScriptedOutbound(Confirmation.ALWAYS)
        ctx = ToolContext(out=outbound)

        first = await manager.call(make_call("weather"), state, ctx)
        second = await manager.call(make_call("weather"), state, ctx)

        assert first == second == message("weather", "weather ran")
        assert len(outbound.requests) == 1
        assert state.always_allowed == {"weather"}
        assert manager.needs_confirmation(manager.new_state(), "weather") is True

    async def test_always_given_in_a_batch_covers_the_later_calls_of_that_tool(self) -> None:
        manager = ToolManager([NamedTool("weather")], confirm=("weather",))
        outbound = ScriptedOutbound(Confirmation.ALWAYS)

        results = await manager.calls(
            [make_call("weather"), make_call("weather")], manager.new_state(), ToolContext(out=outbound)
        )

        assert results == [message("weather", "weather ran")] * 2
        assert len(outbound.requests) == 1

    async def test_always_only_covers_the_tool_it_was_given_for(self) -> None:
        manager = ToolManager([NamedTool("weather"), NamedTool("echo")], confirm=("weather", "echo"))
        state = manager.new_state()
        ctx = ToolContext(out=ScriptedOutbound(Confirmation.ALWAYS, Confirmation.DENY))

        await manager.call(make_call("weather"), state, ctx)

        assert await manager.call(make_call("echo"), state, ctx) == declined("echo")

    async def test_ungated_tool_never_asks(self) -> None:
        outbound = ScriptedOutbound()
        manager = ToolManager([NamedTool("weather"), NamedTool("echo")], confirm=("echo",))

        assert await call(manager, make_call("weather"), outbound) == message("weather", "weather ran")
        assert outbound.requests == []

    async def test_tool_requiring_confirmation_is_gated_without_settings(self) -> None:
        manager = ToolManager([GatedTool("writer")])

        assert manager.needs_confirmation(manager.new_state(), "writer") is True
        assert await call(manager, make_call("writer"), ScriptedOutbound(Confirmation.DENY)) == declined("writer")

    async def test_client_that_cannot_be_asked_declines(self) -> None:
        result = await call(ToolManager([GatedTool("writer")]), make_call("writer"), ClosedOutbound())

        assert result == declined("writer")

    async def test_disabled_gated_tool_is_unknown_without_asking(self) -> None:
        outbound = ScriptedOutbound()

        result = await call(ToolManager([GatedTool("writer")], disabled=("writer",)), make_call("writer"), outbound)

        assert result == message("writer", "error: unknown tool 'writer'")
        assert outbound.requests == []

    def test_unknown_tool_does_not_need_confirmation(self) -> None:
        manager = ToolManager(confirm=("ghost",))

        assert manager.needs_confirmation(manager.new_state(), "ghost") is False

    async def test_missing_arguments_reach_the_client_as_empty(self) -> None:
        outbound = ScriptedOutbound(Confirmation.ALLOW)
        manager = ToolManager([NamedTool("weather")], confirm=("weather",))

        await call(manager, {"type": "function", "function": {"name": "weather", "arguments": None}}, outbound)

        assert outbound.requests == [("weather", {})]


class TestCalls:
    async def test_no_call_gives_no_message(self) -> None:
        manager = make_manager()

        assert await manager.calls([], manager.new_state(), ToolContext(out=RecordingOutbound())) == []

    async def test_calls_run_in_parallel(self) -> None:
        first, second = asyncio.Event(), asyncio.Event()
        manager = ToolManager([HandshakeTool("a", first, second), HandshakeTool("b", second, first)])

        async with asyncio.timeout(1):
            results = await manager.calls(
                [make_call("a"), make_call("b")], manager.new_state(), ToolContext(out=RecordingOutbound())
            )

        assert results == [message("a", "a ran"), message("b", "b ran")]

    async def test_results_keep_the_order_of_the_calls(self) -> None:
        manager = ToolManager([EchoTool(), BrokenTool(), GatedTool("writer"), NamedTool("weather")])
        ctx = ToolContext(out=ScriptedOutbound(Confirmation.DENY))

        results = await manager.calls(
            [
                make_call("weather"),
                make_call("ghost"),
                make_call("writer"),
                make_call("broken"),
                "malformed",
                make_call("echo", {"city": "Paris"}),
            ],
            manager.new_state(),
            ctx,
        )

        assert results == [
            message("weather", "weather ran"),
            message("ghost", "error: unknown tool 'ghost'"),
            declined("writer"),
            message("broken", "error: ValueError: page could not be parsed"),
            message("unknown", "error: malformed tool call, missing function name"),
            message("echo", "Paris"),
        ]

    async def test_a_failing_call_does_not_stop_the_others(self) -> None:
        manager = ToolManager([BrokenTool(), NamedTool("weather")])

        results = await manager.calls(
            [make_call("broken"), make_call("weather")], manager.new_state(), ToolContext(out=RecordingOutbound())
        )

        assert results[1] == message("weather", "weather ran")

    async def test_approvals_are_asked_one_after_another_in_order(self) -> None:
        manager = ToolManager([GatedTool("a"), GatedTool("b"), GatedTool("c")])
        outbound = ScriptedOutbound(Confirmation.ALLOW, Confirmation.DENY, Confirmation.ALLOW, delay=0.01)

        results = await manager.calls(
            [make_call("a"), make_call("b"), make_call("c")], manager.new_state(), ToolContext(out=outbound)
        )

        assert [name for name, _ in outbound.requests] == ["a", "b", "c"]
        assert outbound.overlapped is False
        assert results == [message("a", "a ran"), declined("b"), message("c", "c ran")]

    async def test_cancelling_calls_cancels_the_running_tools(self) -> None:
        first, second = BlockingTool("a"), BlockingTool("b")
        manager = ToolManager([first, second])
        task = asyncio.create_task(
            manager.calls([make_call("a"), make_call("b")], manager.new_state(), ToolContext(out=RecordingOutbound()))
        )
        await first.started.wait()
        await second.started.wait()

        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert first.cancelled
        assert second.cancelled
