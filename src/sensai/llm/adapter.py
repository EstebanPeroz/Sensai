from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, overload

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sensai.llm.responses import ChatResponse, ShowResponse


class InterfaceAdapter(ABC):
    """Interface Used to communicate with different llm API."""

    @overload
    def chat(self, payload: dict, *, stream: Literal[True]) -> Iterator[ChatResponse] | None: ...
    @overload
    def chat(self, payload: dict, *, stream: Literal[False]) -> ChatResponse | None: ...

    @abstractmethod
    def chat(self, payload: dict, *, stream: bool) -> ChatResponse | Iterator[ChatResponse] | None:
        """Send a chat call.

        When stream is True, returns an iterator yielding one ChatResponse per
        chunk (the last one has done=True) instead of a single ChatResponse.
        """

    @abstractmethod
    def embedding(self, message: str, model: str) -> list:
        """Send a message to embed with model to receive a list of vector."""

    @abstractmethod
    def embeddings(self, messages: list[str], model: str) -> list[list]:
        """Send messages to embed with model and receive a list of vector for each message."""

    @abstractmethod
    def available_models(self) -> list[str]:
        """List the models the provider can run."""

    @abstractmethod
    def get_default_model(self) -> str | None:
        """Return the model to use when none is selected, or None when the provider has no model."""

    @abstractmethod
    def show(self, model_name: str) -> ShowResponse | None:
        """Get info on a specified model."""

    @abstractmethod
    def load(self, model_name: str) -> bool:
        """Load a model."""

    @abstractmethod
    def unload(self, model_name: str) -> bool:
        """Load a model."""
