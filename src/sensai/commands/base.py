from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from sensai.ui.adapter import UIAdapter


class Command(ABC):
    """Command class abstract base class for all commands."""

    name: ClassVar[str]
    help: ClassVar[str]
    _ui: UIAdapter

    def __init__(self, ui: UIAdapter) -> None:
        """Initialize the command with the UI it reports to."""
        self._ui = ui

    @abstractmethod
    def execute(self, *args: str) -> None:
        """Execute the command with the given arguments."""
