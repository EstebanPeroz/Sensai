from typing import TYPE_CHECKING, override

from sensai_server.commands._base import Command

if TYPE_CHECKING:
    from sensai_server.session import Session


class Model(Command):
    """Model command class to display or choose available models."""

    name = "model"
    help = "List the available models, or set the model to use."
    usage = "/model [model_name]"

    async def _display_available_models(self, session: Session) -> None:
        """Display the available models."""
        models = session.provider_manager.get_models()
        if not models:
            await session.out.send_system_message("No models are available.")
            return
        lines = [f"Current model: {session.conv.model}", "Available models:"]
        lines.extend(f"- {model}" for model in sorted(models))
        await session.out.send_system_message("\n".join(lines))

    async def _set_model(self, session: Session, model_name: str) -> None:
        """Set the model `session` uses."""
        if await session.set_model(model_name):
            await session.out.send_system_message(f"Model set to '{model_name}'.")
        else:
            await session.out.send_system_message(f"Model '{model_name}' is not available.")

    @override
    async def execute(self, session: Session, *args: str) -> None:
        """Display the available models, or set the model to use."""
        if len(args) == 0:
            await self._display_available_models(session)
            return
        if len(args) == 1:
            await self._set_model(session, args[0])
            return
        await session.out.send_system_message(f"Invalid number of arguments. Usage: {self.usage}")

    @override
    def completions(self, session: Session, args: list[str]) -> list[str]:
        """Return the available model names."""
        del args
        return sorted(session.provider_manager.get_models())
