from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sensai.config.settings import Settings
    from sensai.llm.embedder import Embedder
    from sensai.tools.tool import Tool


def build_tools(settings: Settings, embedder: Embedder | None) -> list[Tool]:
    """Build every tool the model may be given. Single entry point for the composition root.

    Each tool reads its own part of `settings`. The web tools rank fetched pages with `embedder`, so they are left out
    without one. Concrete tools are imported inside this function, so a tool module can import `sensai.tools` without
    a circular import.
    """
    from sensai.tools.web import web_tools  # noqa: PLC0415

    return web_tools(settings.web, embedder) if embedder is not None else []
