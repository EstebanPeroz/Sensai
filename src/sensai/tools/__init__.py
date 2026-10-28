from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from sensai.config.settings import Settings
    from sensai.llm.embedder import Embedder
    from sensai.tools.tool import Tool


def build_tools(settings: Settings, embedder: Embedder | None, ask_user: Callable[[str], str | None]) -> list[Tool]:
    """Build every built-in tool the model may be given. Single entry point for the composition root.

    Each tool reads its own part of `settings`. The web tools rank fetched pages with `embedder`, so they are left out
    without one. `ask_user` puts a question to the user and blocks until they answer (None when they do not). Concrete
    tools are imported inside this function, so a tool module can import `sensai.tools` without a circular import.
    """
    from sensai.tools.ask_user import AskUserTool  # noqa: PLC0415
    from sensai.tools.web import web_tools  # noqa: PLC0415

    tools = web_tools(settings.web, embedder) if embedder is not None else []
    tools.append(AskUserTool(ask_user))
    return tools
