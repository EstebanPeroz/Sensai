from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sensai.llm.responses import ChatResponse


class EventType(Enum):
    """Kind of event sent by a UI: raw user text, or an issued command."""

    UserContent = 1
    Command = 2


class Event:
    """Data sent by a UI when the user performs an action, such as submitting text or issuing a command."""

    type: EventType | None = None
    content: str = ""


class UIAdapter(ABC):
    """Base class a concrete UI implementation must subclass to be driven by the rest of the application.

    Decouples the application logic from any specific UI framework: the
    application pushes AI responses/user input to the adapter and pulls
    user-triggered events from it, without knowing how the UI is implemented.
    """

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
        """Block until an event is available in the queue, or until `timeout` seconds elapse.

        Returns True if an event became available, False on timeout.
        """

    @abstractmethod
    def get_event(self) -> Event | None:
        """Pop and return the oldest pending event, or None if the queue is empty."""

    @abstractmethod
    def send_ai_response(self, response: ChatResponse) -> bool:
        """Forward an AI response to the UI for display. Returns whether it was delivered."""

    @abstractmethod
    def send_user_input(self, response: str) -> bool:
        """Forward a user's conversation input to the UI for display. Returns whether it was delivered."""

    @abstractmethod
    def send_system_message(self, message: str) -> bool:
        """Forward an application message, such as a command output, to the UI. Returns whether it was delivered."""
