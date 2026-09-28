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
    def open() -> bool:
        """Start the UI."""

    @abstractmethod
    def close() -> bool:
        """Stop the UI."""

    @abstractmethod
    def wait_event() -> None:
        """Wait the UI as long that no event is made."""

    @abstractmethod
    def get_event() -> Event | None:
        """Wait the UI as long that no event is made."""

    @abstractmethod
    def send_ai_response() -> ChatResponse:
        """Send a ai response to the UI."""
