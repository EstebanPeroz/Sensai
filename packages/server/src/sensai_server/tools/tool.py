from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sensai_protocol.error import SensaiError

if TYPE_CHECKING:
    from sensai_server.app.outbound import Outbound


def get_parameter[T](param: str, expected_type: type[T], **kwargs: object) -> T:
    """Get a parameter in a keyword args list, raising if its type doesn't match."""
    if param not in kwargs:
        msg = f"Miss parameter: {param}"
        raise SensaiError(msg)

    value = kwargs[param]
    if not isinstance(value, expected_type):
        msg = f"Parameter '{param}' must be {expected_type.__name__}, got {type(value).__name__}"
        raise SensaiError(msg)

    return value


@dataclass(frozen=True)
class ToolContext:
    """What a tool needs from the session that called it: `out`, the outbound of that session's client."""

    out: Outbound


class Tool[T](ABC):
    """Base class every tool must implement.

    One instance is shared by every session: a tool keeps no per-client state, and reaches the calling client
    through the `ToolContext` given to `run`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name used to look up and call this tool."""

    @property
    @abstractmethod
    def description(self) -> str:
        """What the tool does, read by the model to decide when to call it."""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON schema of the arguments accepted by `run`."""

    @property
    def tool_rule(self) -> dict:
        """Describe this tool in the format expected by the chat endpoint."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    @property
    def requires_confirmation(self) -> bool:
        """Whether the user must approve each call, for a tool with consequences such as writing a file."""
        return False

    @abstractmethod
    async def run(self, ctx: ToolContext, **kwargs: object) -> T:
        """Execute the tool for the session of `ctx` and return its result.

        Blocking work (HTTP, embeddings, DB) goes through `asyncio.to_thread`, as rule 4 of `pipeline/chain.py`
        says, so the event loop shared by every session never blocks.
        """
