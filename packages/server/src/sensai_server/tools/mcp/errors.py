from __future__ import annotations

from sensai_protocol.error import SensaiError


def describe(err: BaseException) -> str:
    """Describe `err` by its message, or its type when it has none, and an exception group by its first failure."""
    while isinstance(err, BaseExceptionGroup):
        err = next((sub for sub in err.exceptions if isinstance(sub, Exception)), err.exceptions[0])
    return str(err) or type(err).__name__


class MCPError(SensaiError):
    """Base class for errors raised by Sensai's MCP integration."""


class MCPStartupError(MCPError):
    """Raised when an MCP server cannot be started or initialized."""

    @classmethod
    def from_error(cls, err: BaseException, timeout: float) -> MCPStartupError:
        """Describe `err` as a startup failure, a timeout, whose message is empty, by the time waited."""
        if isinstance(err, TimeoutError):
            return cls(f"no answer within {timeout:g}s")
        return cls(describe(err))


class MCPProtocolError(MCPError):
    """Raised when an MCP server returns an invalid or unexpected response."""


class MCPCallError(MCPError):
    """Raised when an MCP tool call fails."""


class MCPShutdownError(MCPError):
    """Raised when an MCP server cannot be shut down cleanly."""
