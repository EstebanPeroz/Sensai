import re
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit

# The name prefixes the server's tool names, so it must also be valid in an LLM function name.
_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")

Transport = Literal["stdio", "http"]


@dataclass
class MCPServerConfig:
    """Configuration needed to start and call one MCP server.

    A stdio server is a subprocess started from `command` and `args`; an http server is a remote one
    reached at `url`, with `headers` sent on every request, such as an authorization token.
    """

    name: str
    transport: Transport = "stdio"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    startup_timeout: float = 10.0
    call_timeout: float = 30.0

    def __post_init__(self) -> None:
        """Validate values that would make an MCP server unusable."""
        if not _NAME_PATTERN.fullmatch(self.name):
            msg = "MCP server name must be non-empty and only contain letters, digits, '_' or '-'"
            raise ValueError(msg)
        if self.transport == "stdio":
            self._check_stdio()
        elif self.transport == "http":
            self._check_http()
        else:
            msg = f"MCP server transport must be 'stdio' or 'http', got {self.transport!r}"
            raise ValueError(msg)
        if self.startup_timeout <= 0:
            msg = "MCP server startup timeout must be greater than zero"
            raise ValueError(msg)
        if self.call_timeout <= 0:
            msg = "MCP server call timeout must be greater than zero"
            raise ValueError(msg)

    def _check_stdio(self) -> None:
        if not self.command:
            msg = "MCP server command must be a non-empty string"
            raise ValueError(msg)
        if self.url is not None or self.headers:
            msg = "MCP server url and headers only apply to the http transport"
            raise ValueError(msg)

    def _check_http(self) -> None:
        parts = urlsplit(self.url or "")
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            msg = f"MCP server url must be an http(s) URL with a host, got {self.url!r}"
            raise ValueError(msg)
        if self.command is not None or self.args:
            msg = "MCP server command and args only apply to the stdio transport"
            raise ValueError(msg)
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in self.headers.items()):
            msg = "MCP server headers must map strings to strings"
            raise ValueError(msg)
