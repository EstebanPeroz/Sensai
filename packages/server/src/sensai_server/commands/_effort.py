from typing import TYPE_CHECKING, override

from sensai_server.commands._base import Command

if TYPE_CHECKING:
    from sensai_server.session import Session

EFFORTS = {"off": 0, "low": 1, "medium": 3, "high": 5}


class Effort(Command):
    """Effort command class to display or choose how many thinking steps the model may take before answering."""

    name = "effort"
    help = "Show the thinking effort, or set it to off, low, medium or high."
    usage = "/effort [off|low|medium|high]"

    async def _display_effort(self, session: Session) -> None:
        """Display the current effort of `session` and the available ones."""
        lines = [f"Current effort: {session.effort}", "Available efforts:"]
        lines.extend(f"- {name}: {self._describe(steps)}" for name, steps in EFFORTS.items())
        await session.out.send_system_message("\n".join(lines))

    async def _set_effort(self, session: Session, level: str) -> None:
        """Set the effort of `session` to `level`."""
        if level not in EFFORTS:
            await session.out.send_system_message(f"Effort '{level}' does not exist. Usage: {self.usage}")
            return
        session.effort = level
        await session.out.update_thinking_effort(level)
        await session.out.send_system_message(f"Effort set to '{level}': {self._describe(EFFORTS[level])}.")

    @staticmethod
    def _describe(steps: int) -> str:
        """Describe what an effort of `steps` thinking steps does."""
        if steps == 0:
            return "answer directly, without thinking nor tools"
        return f"up to {steps} thinking steps"

    @override
    async def execute(self, session: Session, *args: str) -> None:
        """Display the current effort, or set the effort to use."""
        if len(args) == 0:
            await self._display_effort(session)
            return
        if len(args) == 1:
            await self._set_effort(session, args[0])
            return
        await session.out.send_system_message(f"Invalid number of arguments. Usage: {self.usage}")

    @override
    def completions(self, session: Session, args: list[str]) -> list[str]:
        """Return the available efforts."""
        del session, args
        return list(EFFORTS)
