from __future__ import annotations

from sensai.error import SensaiError


class MCPError(SensaiError):
    """Base class for errors raised by Sensai's MCP integration."""


class MCPStartupError(MCPError):
    """Raised when an MCP server cannot be started or initialized."""

    @classmethod
    def from_error(cls, err: BaseException, timeout: float) -> MCPStartupError:
        """Describe `err` as a startup failure, by its first failure when it is an exception group.

        A timeout, whose message is empty, is described by the time waited.
        """
        while isinstance(err, BaseExceptionGroup):
            err = next((sub for sub in err.exceptions if isinstance(sub, Exception)), err.exceptions[0])
        if isinstance(err, TimeoutError):
            return cls(f"no answer within {timeout:g}s")
        return cls(str(err) or type(err).__name__)


class MCPProtocolError(MCPError):
    """Raised when an MCP server returns an invalid or unexpected response."""


class MCPCallError(MCPError):
    """Raised when an MCP tool call fails."""


class MCPShutdownError(MCPError):
    """Raised when an MCP server cannot be shut down cleanly."""
