from sensai.error import SensaiError


class MCPError(SensaiError):
    """Base class for errors raised by Sensai's MCP integration."""


class MCPStartupError(MCPError):
    """Raised when an MCP server cannot be started or initialized."""


class MCPProtocolError(MCPError):
    """Raised when an MCP server returns an invalid or unexpected response."""


class MCPCallError(MCPError):
    """Raised when an MCP tool call fails."""


class MCPShutdownError(MCPError):
    """Raised when an MCP server cannot be shut down cleanly."""
