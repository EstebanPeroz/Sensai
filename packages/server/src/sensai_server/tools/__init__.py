"""The tools the model can call, shared by every session."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sensai_server.tools.ask_user import AskUserTool
from sensai_server.tools.mcp import MCPHost
from sensai_server.tools.web import web_tools

if TYPE_CHECKING:
    from sensai_server.config.settings import Settings
    from sensai_server.llm.embedder import Embedder
    from sensai_server.tools.tool import Tool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuiltTools:
    """Every tool `build_tools` built, and the MCP host serving the MCP ones, for the app to close at shutdown."""

    tools: list[Tool]
    mcp: MCPHost


def build_tools(settings: Settings, embedder: Embedder | None) -> BuiltTools:
    """Build every tool the model may be given, built-in or MCP. Single entry point for the composition root.

    Each tool reads its own part of `settings`. The web tools rank fetched pages with `embedder`, so they are left out
    without one. The MCP host is started here: a server failing to start is logged, as there is no client to tell yet,
    and its tools are left out. The caller owns the returned host and must close it. Concrete tools are imported
    inside this function, so a tool module can import `sensai_server.tools` without a circular import.
    """
    tools: list[Tool] = web_tools(settings.web, embedder) if embedder is not None else []
    tools.append(AskUserTool())
    host = MCPHost(settings.mcp)
    mcp_tools, failures = host.start()
    for name, err in failures.items():
        logger.warning("MCP server '%s' failed to start, its tools are unavailable: %s", name, err)
    tools.extend(mcp_tools)
    return BuiltTools(tools=tools, mcp=host)
