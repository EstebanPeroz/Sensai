from dataclasses import dataclass


@dataclass(frozen=True)
class MCPToolInfo:
    """Small internal description of a tool advertised by an MCP server."""

    name: str
    description: str | None
    input_schema: dict[str, object]
