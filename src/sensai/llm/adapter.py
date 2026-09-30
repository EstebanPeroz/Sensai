from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, overload

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sensai.llm.responses import ChatResponse, ShowResponse


class ProviderAdapter(ABC):
    """Interface Used to communicate with different llm API."""

    @overload
    def chat(self, payload: dict, *, stream: Literal[True]) -> Iterator[ChatResponse]: ...
    @overload
    def chat(self, payload: dict, *, stream: Literal[False]) -> ChatResponse: ...

    @abstractmethod
    def chat(self, payload: dict, *, stream: bool) -> ChatResponse | Iterator[ChatResponse]:
        """Send a chat call.

        When stream is True, returns an iterator yielding one ChatResponse per
        chunk (the last one has done=True) instead of a single ChatResponse.
        """

    @abstractmethod
    def embedding(self, message: str) -> list:
        """Send a message to embed to receive a list of vector."""

    @abstractmethod
    def embeddings(self, messages: list[str]) -> list[list]:
        """Send messages to embed and receive a list of vector for each message."""

    @abstractmethod
    def show(self, model_name: str) -> ShowResponse:
        """Get info on a specified model."""

    @abstractmethod
    def load(self, model_name: str) -> bool:
        """Load a model."""

    @abstractmethod
    def unload(self, model_name: str) -> bool:
        """Load a model."""

    @abstractmethod
    def list(self) -> list[str]:
        """List of model given by the provider."""
