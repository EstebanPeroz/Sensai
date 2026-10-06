import pytest

from sensai.error import SensaiError
from sensai.tools.tool import Tool, get_parameter


class DummyTool(Tool[str]):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def tool_rule(self) -> dict:
        return {"type": "function"}

    def run(self, **kwargs: object) -> str:
        return str(kwargs.get("value"))


class MultiParamTool(Tool[str]):
    @property
    def name(self) -> str:
        return "multi"

    @property
    def tool_rule(self) -> dict:
        return {"type": "function"}

    def run(self, **kwargs: object) -> str:
        city = get_parameter("city", str, **kwargs)
        age = get_parameter("age", int, **kwargs)
        return f"{city}-{age}"


class IncompleteTool(Tool[None]):
    @property
    def name(self) -> str:
        return "incomplete"

    def run(self, **kwargs: object) -> None:
        del kwargs


class TestToolAbstractness:
    def test_cannot_instantiate_tool_directly(self) -> None:
        with pytest.raises(TypeError):
            Tool()

    def test_subclass_missing_members_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            IncompleteTool()

    def test_full_subclass_can_be_instantiated_and_used(self) -> None:
        tool = DummyTool()

        assert tool.name == "dummy"
        assert tool.tool_rule == {"type": "function"}
        assert tool.run(value=42) == "42"


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
    def test_run_reads_multiple_parameters_from_kwargs(self) -> None:
        tool = MultiParamTool()

        assert tool.run(city="Paris", age=30, active=True) == "Paris-30"

    def test_run_ignores_unrelated_extra_kwargs(self) -> None:
        tool = MultiParamTool()

        assert tool.run(city="Paris", age=30, extra_field="ignored", another=123) == "Paris-30"

    def test_run_raises_when_one_of_many_parameters_has_wrong_type(self) -> None:
        tool = MultiParamTool()

        with pytest.raises(SensaiError, match="Parameter 'age' must be int, got str"):
            tool.run(city="Paris", age="old", active=True)
