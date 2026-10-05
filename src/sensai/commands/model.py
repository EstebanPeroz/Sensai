from typing import TYPE_CHECKING, override

from sensai.commands.base import Command

if TYPE_CHECKING:
    from sensai.llm.provider_manager import ProviderManager
    from sensai.ui.adapter import UIAdapter


class Model(Command):
    """Model command class to display or choose available models."""

    name = "model"
    help = "List the available models, or set the model to use."
    usage = "/model [model_name]"
    _provider_manager: ProviderManager

    def __init__(self, ui: UIAdapter, provider_manager: ProviderManager) -> None:
        """Initialize the Model command with the provider manager serving the models."""
        super().__init__(ui)
        self._provider_manager = provider_manager

    def _display_available_models(self) -> None:
        """Display the available models."""
        models = self._provider_manager.get_models()
        if not models:
            self._ui.send_system_message("No models are available.")
            return
        current_model = self._provider_manager.provider().current_model()
        lines = [f"Current model: {current_model}"]
        lines.append("Available models:")
        lines.extend(f"- {model}" for model in sorted(models))
        self._ui.send_system_message("\n".join(lines))

    def _set_model(self, model_name: str) -> None:
        """Set the model to use."""
        self._ui.send_system_message(f"Setting model to '{model_name}'...")
        if self._provider_manager.set_model(model_name):
            self._ui.send_system_message(f"Model set to '{model_name}'.", append_response=True)
        else:
            self._ui.send_system_message(f"Model '{model_name}' is not available.")

    @override
    def execute(self, *args: str) -> None:
        """Display the available models, or set the model to use."""
        if len(args) == 0:
            self._display_available_models()
            return
        if len(args) == 1:
            self._set_model(args[0])
            return
        self._ui.send_system_message(f"Invalid number of arguments. Usage: {self.usage}")

    @override
    def completions(self) -> list[str]:
        """Return the available model names."""
        return sorted(self._provider_manager.get_models())
