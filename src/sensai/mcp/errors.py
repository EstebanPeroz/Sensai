from __future__ import annotations

from sensai.error import SensaiError


class MCPError(SensaiError):
    """Base class for errors raised by Sensai's MCP integration."""


class MCPStartupError(MCPError):
    """Raised when an MCP server cannot be started or initialized."""

    @classmethod
    def from_error(cls, err: Exception, timeout: float) -> MCPStartupError:
        """Describe `err`, whose message is empty when it is a timeout, as a startup failure."""
        if isinstance(err, TimeoutError):
            return cls(f"no answer within {timeout:g}s")
        return cls(str(err) or type(err).__name__)


class MCPProtocolError(MCPError):
    """Raised when an MCP server returns an invalid or unexpected response."""


class MCPCallError(MCPError):
    """Raised when an MCP tool call fails."""


class MCPShutdownError(MCPError):
    """Raised when an MCP server cannot be shut down cleanly."""
