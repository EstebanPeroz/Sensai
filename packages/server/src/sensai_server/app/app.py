from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from sensai_protocol import (
    Cancel,
    Chat,
    Command,
    CommandContext,
    Complete,
    ConfirmAnswer,
    Login,
    Logout,
    ProtocolError,
    UserAnswer,
    parse_client_message,
)

from sensai_server.app import _handlers as handlers
from sensai_server.app._state import AppState, app_state, set_app_state
from sensai_server.app.outbound import Outbound
from sensai_server.commands import CommandRegistry, build_commands
from sensai_server.llm.embedder import Embedder
from sensai_server.llm.persona import DuplicatePersonaError, PersonaManager
from sensai_server.llm.provider import ProviderManager
from sensai_server.memory.persistent_db.base import get_db, set_db
from sensai_server.memory.persistent_db.error import DbConnectionError
from sensai_server.pipeline import ChatPipeline
from sensai_server.pipeline.modules import ModelStage, PersonaStage
from sensai_server.pipeline.modules.guardrails import BadWordsGuard
from sensai_server.session import SessionRegistry
from sensai_server.tools import build_tools
from sensai_server.tools.tool_manager import ToolManager

if TYPE_CHECKING:
    from sensai_protocol import ClientMessage

    from sensai_server.config.settings import Settings
    from sensai_server.session.session import Session

logger = logging.getLogger(__name__)

router = APIRouter()


def create_app(
    settings: Settings,
) -> FastAPI:
    """Build a new app serving the `/ws` route, logging clients in on the models of the configured providers.

    `settings` are the loaded settings, kept in the app state (see `app_state`). The persistent database is opened,
    and the default persona is stored in it if it is not there yet. The chat pipeline is built once
    (see `_build_pipeline`) and shared by every session: each `chat` prompt goes through it as the session's task. So
    is the tool manager (see `_build_tool_manager`): each session keeps only its own tool state.
    """
    pm: ProviderManager = ProviderManager(settings.llm)
    embedder = _build_embedder(settings, pm)
    tool_manager = _build_tool_manager(settings, embedder)
    pipeline = _build_pipeline(pm, tool_manager)
    set_db(settings.db.persistent)
    db = get_db()
    if db is None:
        msg = "the persistent database is not open"
        raise DbConnectionError(msg)
    persona_manager = PersonaManager(db)
    with contextlib.suppress(DuplicatePersonaError):
        persona_manager.create_default_persona()

    sessions = SessionRegistry(pm, persona_manager, tool_manager)
    commands = CommandRegistry()
    commands.set_commands(build_commands(sessions, commands, db, pm, tool_manager))

    app = FastAPI()
    set_app_state(
        app,
        AppState(
            settings=settings,
            provider_manager=pm,
            embedder=embedder,
            tool_manager=tool_manager,
            db=db,
            persona_manager=persona_manager,
            registry=sessions,
            commands=commands,
            pipeline=pipeline,
        ),
    )
    app.include_router(router)
    return app


def _build_pipeline(pm: ProviderManager, tool_manager: ToolManager) -> ChatPipeline:
    """Build the chain every chat goes through: the guard first, then the persona, then the model."""
    return ChatPipeline(ModelStage(pm, tool_manager)).use(BadWordsGuard()).use(PersonaStage())


def _build_embedder(settings: Settings, pm: ProviderManager) -> Embedder | None:
    """Build the embedder of the web tools, or None without an embedding model or a provider serving it.

    Without it, the web tools are left out: at startup there is no client to tell, so a missing provider is logged.
    """
    model = settings.llm.embedding_model
    if model is None:
        return None
    provider = pm.get_provider(model)
    if provider is None:
        logger.warning("no provider serves the embedding model '%s': the web tools are left out", model)
        return None
    return Embedder(provider, model)


def _build_tool_manager(settings: Settings, embedder: Embedder | None) -> ToolManager:
    """Build the manager of every built-in tool, shared by every session.

    Names in the disabled or confirm settings matching no tool are logged: at startup there is no client to tell.
    """
    manager = ToolManager(
        build_tools(settings, embedder),
        disabled=settings.features.disabled_in("tools"),
        confirm=settings.features.confirm_in("tools"),
    )
    for name in manager.unknown_disabled():
        logger.warning("unknown tool '%s' in the disabled features", name)
    for name in manager.unknown_confirm():
        logger.warning("unknown tool '%s' in the confirm features", name)
    return manager


@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    """Receive the client's messages and dispatch each one to its handler, until logout or disconnection.

    A frame that is not a valid client message, or any message but `login` before a session is open, is answered
    with an `error`, and the connection stays open. However the connection ends, its session is freed.
    """
    await ws.accept()
    state = app_state(ws.app)
    out = Outbound(ws)
    session: Session | None = None
    try:
        async for raw in ws.iter_text():
            try:
                message = parse_client_message(raw)
            except ProtocolError as err:
                await out.send_error(str(err))
                continue
            if isinstance(message, Login):
                session = await handlers.handle_login(state.registry, out, session, message, state.commands.names())
                continue
            if session is None:
                await out.send_error("Login first.")
                continue
            if not await _dispatch(ws, state, session, message):
                break
    except WebSocketDisconnect:
        pass
    finally:
        if session is not None:
            await state.registry.close(session.client_id)


async def _dispatch(ws: WebSocket, state: AppState, session: Session, message: ClientMessage) -> bool:
    """Dispatch a message, other than `login`, of a logged-in client to its handler.

    Returns False once the connection must end.
    """
    match message:
        case Logout():
            await handlers.handle_logout(state.registry, session, message)
            await ws.close()
            return False
        case Chat():
            await handlers.handle_chat(session, message, state.pipeline, state.db)
        case Command():
            await handlers.handle_command(session, message, state.commands)
        case Cancel():
            await handlers.handle_cancel(session, message)
        case Complete():
            await handlers.handle_complete(session, message, state.commands)
        case ConfirmAnswer():
            await handlers.handle_confirm_answer(session, message)
        case UserAnswer():
            await handlers.handle_user_answer(session, message)
        case CommandContext():
            await handlers.handle_command_context(session, message)
    return True
