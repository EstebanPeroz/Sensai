from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from sensai_server.cache import SemanticCache
    from sensai_server.chat.react import ReAct
    from sensai_server.commands import CommandRegistry
    from sensai_server.config.settings import Settings
    from sensai_server.events import EventBus
    from sensai_server.llm.embedder import Embedder
    from sensai_server.llm.persona import PersonaManager
    from sensai_server.llm.provider import ProviderManager
    from sensai_server.memory.persistent_db.adapter import PersistentDatabase
    from sensai_server.monitoring.logger import InteractionLogger
    from sensai_server.pipeline import ChatPipeline
    from sensai_server.rag import Rag
    from sensai_server.session import SessionRegistry
    from sensai_server.tools.mcp import MCPHost
    from sensai_server.tools.tool_manager import ToolManager

# The attribute of `app.state` holding the `AppState`.
STATE_KEY = "sensai"


@dataclass(frozen=True)
class AppState:
    """Everything the routes share, built once by `create_app`.

    Starlette's `app.state` types its attributes as `Any`: the app keeps this one typed object in it instead, read
    with `app_state`.
    """

    settings: Settings
    provider_manager: ProviderManager
    embedder: Embedder | None
    tool_manager: ToolManager
    react: ReAct
    db: PersistentDatabase
    persona_manager: PersonaManager
    registry: SessionRegistry
    commands: CommandRegistry
    pipeline: ChatPipeline
    rag: Rag | None
    cache: SemanticCache | None
    events: EventBus
    interaction_logger: InteractionLogger
    mcp: MCPHost


def set_app_state(app: Starlette, state: AppState) -> None:
    """Keep `state` in `app`, for the routes to read with `app_state`."""
    setattr(app.state, STATE_KEY, state)


def app_state(app: Starlette) -> AppState:
    """Return the state `create_app` kept in `app`."""
    return getattr(app.state, STATE_KEY)
