from typing import TYPE_CHECKING, override

from sensai.commands.base import Command

if TYPE_CHECKING:
    from sensai.llm.thinking.react import ReAct
    from sensai.ui.adapter import UIAdapter

EFFORTS = {"off": 0, "low": 1, "medium": 3, "high": 5}
LEVELS = {steps: name for name, steps in EFFORTS.items()}


class Effort(Command):
    """Effort command class to display or choose how many thinking steps the model may take before answering."""

    name = "effort"
    help = "Show the thinking effort, or set it to off, low, medium or high."
    usage = "/effort [off|low|medium|high]"
    _react: ReAct

    def __init__(self, ui: UIAdapter, react: ReAct) -> None:
        """Initialize the Effort command with the ReAct loop whose thinking steps it limits."""
        super().__init__(ui)
        self._react = react

    def _display_effort(self) -> None:
        """Display the current effort and the available ones."""
        lines = [f"Current effort: {self.effort_level(self._react.max_iterations)}", "Available efforts:"]
        lines.extend(f"- {name}: {self._describe(steps)}" for name, steps in EFFORTS.items())
        self._ui.send_system_message("\n".join(lines))

    def _set_effort(self, level: str) -> None:
        """Set the effort to `level`."""
        if level not in EFFORTS:
            self._ui.send_system_message(f"Effort '{level}' does not exist. Usage: {self.usage}")
            return
        steps = EFFORTS[level]
        self._react.update_max_iterations(steps)
        self._ui.update_thinking_effort(level)
        self._ui.send_system_message(f"Effort set to '{level}': {self._describe(steps)}.")

    @staticmethod
    def _describe(steps: int) -> str:
        """Describe what an effort of `steps` thinking steps does."""
        if steps == 0:
            return "answer directly, without thinking nor tools"
        return f"up to {steps} thinking steps"

    @override
    def execute(self, *args: str) -> None:
        """Display the current effort, or set the effort to use."""
        if len(args) == 0:
            self._display_effort()
            return
        if len(args) == 1:
            self._set_effort(args[0])
            return
        self._ui.send_system_message(f"Invalid number of arguments. Usage: {self.usage}")

    @override
    def completions(self) -> list[str]:
        """Return the available efforts."""
        return list(EFFORTS)

    @staticmethod
    def effort_level(steps: int) -> str:
        """Return the name of the effort allowing `steps` thinking steps, or the number itself when none does."""
        return LEVELS.get(steps, f"{steps} thinking steps")
