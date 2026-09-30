from abc import ABC, abstractmethod
from typing import ClassVar


class Command(ABC):
    """Command class abstract base class for all commands."""

    name: ClassVar[str]
    help: ClassVar[str]

    @abstractmethod
    def execute(self, *args: str) -> None:
        """Execute the command with the given arguments."""
