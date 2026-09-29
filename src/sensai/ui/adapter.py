from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sensai.llm.responses import ChatResponse


class EventType(Enum):
    """class representing what type of event is sent from UI."""

    UserContent = 1
    Command = 2


class Event:
    """Event sent by a UI."""

    type: EventType | None = None
    content: str = ""


class UIAdapter(ABC):
    """Adapter to use to make a valid UI support."""

    @abstractmethod
    def open(self) -> bool:
        """Wait until the UI has started and is ready to receive calls."""

    @abstractmethod
    def run(self) -> None:
        """Block running the UI. Must be called from the main thread."""

    @abstractmethod
    def close(self) -> bool:
        """Stop the UI."""

    @abstractmethod
    def wait_event(self, *, timeout: float | None = None) -> bool:
        """Wait the UI as long that no event is made."""

    @abstractmethod
    def get_event(self) -> Event | None:
        """Get the first event in the event queue."""

    @abstractmethod
    def send_ai_response(self, response: ChatResponse) -> bool:
        """Send a ai response to the UI."""

    @abstractmethod
    def send_user_input(self, response: str) -> bool:
        """Send a user conversation input to the UI."""

    @abstractmethod
    def send_ai_error(self, response: str) -> bool:
        """Send a ai error to the UI."""
