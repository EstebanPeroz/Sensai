from unittest.mock import MagicMock

import pytest

from sensai.commands.effort import EFFORTS, Effort
from sensai.llm.thinking.react import ReAct
from sensai.tools.tool_manager import ToolManager
from tests.commands.fakes import FakeUI


@pytest.fixture
def react() -> ReAct:
    return ReAct(MagicMock(), ToolManager(), max_iterations=5)


class TestEffort:
    def test_no_arguments_displays_current_and_available_efforts(self, fake_ui: FakeUI, react: ReAct) -> None:
        Effort(react).execute()

        assert fake_ui.messages == [
            (
                "Current effort: high\n"
                "Available efforts:\n"
                "- off: answer directly, without thinking nor tools\n"
                "- low: up to 1 thinking steps\n"
                "- medium: up to 3 thinking steps\n"
                "- high: up to 5 thinking steps"
            )
        ]

    def test_no_arguments_with_effort_outside_the_levels(self, fake_ui: FakeUI) -> None:
        Effort(ReAct(MagicMock(), ToolManager(), max_iterations=7)).execute()

        assert fake_ui.messages[0].startswith("Current effort: 7 thinking steps\n")

    @pytest.mark.parametrize(("level", "steps"), list(EFFORTS.items()))
    def test_one_argument_sets_effort(self, fake_ui: FakeUI, react: ReAct, level: str, steps: int) -> None:
        Effort(react).execute(level)

        assert react.max_iterations == steps
        assert fake_ui.messages[0].startswith(f"Effort set to '{level}': ")

    @pytest.mark.parametrize("level", list(EFFORTS))
    def test_one_argument_shows_effort_in_ui(self, fake_ui: FakeUI, react: ReAct, level: str) -> None:
        Effort(react).execute(level)

        assert fake_ui.thinking_efforts == [level]

    def test_unknown_effort_keeps_current_one(self, fake_ui: FakeUI, react: ReAct) -> None:
        Effort(react).execute("extreme")

        assert react.max_iterations == 5
        assert fake_ui.thinking_efforts == []
        assert fake_ui.messages == ["Effort 'extreme' does not exist. Usage: /effort [off|low|medium|high]"]

    def test_too_many_arguments_displays_usage(self, fake_ui: FakeUI, react: ReAct) -> None:
        Effort(react).execute("low", "high")

        assert react.max_iterations == 5
        assert fake_ui.messages == ["Invalid number of arguments. Usage: /effort [off|low|medium|high]"]

    def test_completions_are_the_efforts(self, react: ReAct) -> None:
        assert Effort(react).completions([]) == ["off", "low", "medium", "high"]


class TestEffortLevel:
    @pytest.mark.parametrize(("steps", "level"), [(0, "off"), (1, "low"), (3, "medium"), (5, "high")])
    def test_names_the_effort_of_each_level(self, steps: int, level: str) -> None:
        assert Effort.effort_level(steps) == level

    def test_gives_the_number_of_steps_outside_the_levels(self) -> None:
        assert Effort.effort_level(7) == "7 thinking steps"
