import re
from dataclasses import dataclass, field

# The name prefixes the server's tool names, so it must also be valid in an LLM function name.
_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


@dataclass
class MCPServerConfig:
    """Configuration needed to start and call one MCP server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    enabled: bool = True
    startup_timeout: float = 10.0
    call_timeout: float = 30.0

    def __post_init__(self) -> None:
        """Validate values that would make an MCP server unusable."""
        if not _NAME_PATTERN.fullmatch(self.name):
            msg = "MCP server name must be non-empty and only contain letters, digits, '_' or '-'"
            raise ValueError(msg)
        if not self.command:
            msg = "MCP server command must be a non-empty string"
            raise ValueError(msg)
        if self.startup_timeout <= 0:
            msg = "MCP server startup timeout must be greater than zero"
            raise ValueError(msg)
        if self.call_timeout <= 0:
            msg = "MCP server call timeout must be greater than zero"
            raise ValueError(msg)
