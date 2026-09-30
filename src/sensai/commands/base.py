from abc import ABC, abstractmethod


class Command(ABC):
    """Command class abstract base class for all commands."""

    @abstractmethod
    def execute(self, *args: str) -> None:
        """Execute the command with the given arguments."""
