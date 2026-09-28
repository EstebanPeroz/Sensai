from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import requests

    from sensai.llm.responses import ShowResponse


class InterfaceAdapter(ABC):
    """Interface Used to communicate with different llm API."""

    @abstractmethod
    def chat(self, payload: dict, *, stream: bool) -> requests.Response:
        """Send a chat call."""

    @abstractmethod
    def embedding(self, message: str) -> list:
        """Send a message to embed to receive a list of vector."""

    @abstractmethod
    def embeddings(self, messages: list[str]) -> list[list]:
        """Send messages to embed and receive a list of vector for each message."""

    @abstractmethod
    def show(self, model_name: str) -> ShowResponse | None:
        """Get info on a specified model."""

    @abstractmethod
    def load(self, model_name: str) -> bool:
        """Load a model."""

    @abstractmethod
    def unload(self, model_name: str) -> bool:
        """Load a model."""
