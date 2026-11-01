import pytest
from sensai_protocol import System
from sensai_protocol.error import SensaiError
from sensai_server.tools.error import DuplicateToolError
from sensai_server.tools.tool import Tool, ToolContext, get_parameter

from tests.server.fakes import RecordingOutbound


class DummyTool(Tool[str]):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "Return the given value as text."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {"value": {"type": "string"}}}

    async def run(self, ctx: ToolContext, **kwargs: object) -> str:
        del ctx
        return str(kwargs.get("value"))


class MultiParamTool(Tool[str]):
    @property
    def name(self) -> str:
        return "multi"

    @property
    def description(self) -> str:
        return "Join a city and an age."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"city": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["city", "age"],
        }

    async def run(self, ctx: ToolContext, **kwargs: object) -> str:
        del ctx
        city = get_parameter("city", str, **kwargs)
        age = get_parameter("age", int, **kwargs)
        return f"{city}-{age}"


class GreetTool(Tool[None]):
    """A tool talking to the client that called it."""

    @property
    def name(self) -> str:
        return "greet"

    @property
    def description(self) -> str:
        return "Greet the user."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def run(self, ctx: ToolContext, **kwargs: object) -> None:
        del kwargs
        await ctx.out.send_system_message("hi")


class IncompleteTool(Tool[None]):
    @property
    def name(self) -> str:
        return "incomplete"

    async def run(self, ctx: ToolContext, **kwargs: object) -> None:
        del ctx, kwargs


@pytest.fixture
def out() -> RecordingOutbound:
    return RecordingOutbound()


@pytest.fixture
def ctx(out: RecordingOutbound) -> ToolContext:
    return ToolContext(out=out)


class TestToolAbstractness:
    def test_cannot_instantiate_tool_directly(self) -> None:
        with pytest.raises(TypeError):
            Tool()

    def test_subclass_missing_members_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            IncompleteTool()

    async def test_full_subclass_can_be_instantiated_and_used(self, ctx: ToolContext) -> None:
        tool = DummyTool()

        assert tool.name == "dummy"
        assert tool.tool_rule == {
            "type": "function",
            "function": {"name": "dummy", "description": tool.description, "parameters": tool.parameters},
        }
        assert await tool.run(ctx, value=42) == "42"


class TestToolContext:
    async def test_tool_reaches_the_calling_client_through_ctx(self, ctx: ToolContext, out: RecordingOutbound) -> None:
        await GreetTool().run(ctx)

        assert out.sent == [System(message="hi")]


class TestGetParameter:
    def test_returns_value_when_present_and_correct_type(self) -> None:
        assert get_parameter("city", str, city="Paris") == "Paris"

    def test_raises_when_missing(self) -> None:
        with pytest.raises(SensaiError, match="Miss parameter: city"):
            get_parameter("city", str)

    def test_raises_when_wrong_type(self) -> None:
        with pytest.raises(SensaiError, match="Parameter 'city' must be str, got int"):
            get_parameter("city", str, city=2)


class TestGetParameterWithManyValues:
    def test_extracts_the_requested_key_among_many(self) -> None:
        assert get_parameter("city", str, city="Paris", age=30, active=True) == "Paris"

    def test_unrelated_extra_kwargs_are_ignored(self) -> None:
        assert get_parameter("age", int, city="Paris", age=30, active=True) == 30

    def test_each_call_narrows_its_own_key_independently(self) -> None:
        kwargs = {"city": "Paris", "age": 30, "active": True}

        assert get_parameter("city", str, **kwargs) == "Paris"
        assert get_parameter("age", int, **kwargs) == 30
        assert get_parameter("active", bool, **kwargs) is True

    def test_raises_for_the_mismatched_key_even_with_other_valid_keys_present(self) -> None:
        with pytest.raises(SensaiError, match="Parameter 'age' must be int, got str"):
            get_parameter("age", int, city="Paris", age="thirty", active=True)

    def test_raises_for_missing_key_even_when_many_other_keys_present(self) -> None:
        with pytest.raises(SensaiError, match="Miss parameter: country"):
            get_parameter("country", str, city="Paris", age=30, active=True)


class TestToolWithManyKeywordArguments:
    async def test_run_reads_multiple_parameters_from_kwargs(self, ctx: ToolContext) -> None:
        assert await MultiParamTool().run(ctx, city="Paris", age=30, active=True) == "Paris-30"

    async def test_run_ignores_unrelated_extra_kwargs(self, ctx: ToolContext) -> None:
        assert await MultiParamTool().run(ctx, city="Paris", age=30, extra_field="ignored", another=123) == "Paris-30"

    async def test_run_raises_when_one_of_many_parameters_has_wrong_type(self, ctx: ToolContext) -> None:
        with pytest.raises(SensaiError, match="Parameter 'age' must be int, got str"):
            await MultiParamTool().run(ctx, city="Paris", age="old", active=True)


class TestRequiresConfirmation:
    def test_tools_do_not_require_confirmation_by_default(self) -> None:
        assert DummyTool().requires_confirmation is False


class TestDuplicateToolError:
    def test_message_names_the_tool(self) -> None:
        assert str(DuplicateToolError("x")) == "[DuplicateToolError] two tools are registered as 'x'"

    def test_is_the_protocol_sensai_error(self) -> None:
        assert isinstance(DuplicateToolError("x"), SensaiError)
