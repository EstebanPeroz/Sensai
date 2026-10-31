"""The tools the model can call, shared by every session."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sensai_server.tools.ask_user import AskUserTool
from sensai_server.tools.web import web_tools

if TYPE_CHECKING:
    from sensai_server.config.settings import Settings
    from sensai_server.llm.embedder import Embedder
    from sensai_server.tools.tool import Tool


def build_tools(settings: Settings, embedder: Embedder | None) -> list[Tool]:
    """Build every built-in tool the model may be given. Single entry point for the composition root.

    Each tool reads its own part of `settings`. The web tools rank fetched pages with `embedder`, so they are left out
    without one. Concrete tools are imported inside this function, so a tool module can import `sensai_server.tools`
    without a circular import.
    """
    tools = web_tools(settings.web, embedder) if embedder is not None else []
    tools.append(AskUserTool())
    return tools
