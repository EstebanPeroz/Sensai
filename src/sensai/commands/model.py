from typing import override

from sensai.commands.base import Command


class Model(Command):
    """Model command class to display or choose available models."""

    name = "model"
    help = "Set the model to use."
    usage = "/model [model_name]"

    # add provider as parameter when implemented
    def __init__(self) -> None:
        """Initialize the Model command."""

    def _set_model(self, model_name: str) -> None:
        """Set the model to use."""

    @override
    def execute(self, *args: str) -> None:
        """Print the available models."""
        if len(args) == 0:
            # Print the available models
            return
        if len(args) == 1:
            self._set_model(args[0])
            return
        print("Invalid number of arguments. Usage: /model [model_name]")
