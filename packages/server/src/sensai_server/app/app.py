from __future__ import annotations

import contextlib
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from redis import Redis as SyncRedis
from redis.exceptions import RedisError
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
from sensai_server.cache import CacheError, SemanticCache
from sensai_server.chat.react import ReAct
from sensai_server.commands import CommandRegistry, build_commands
from sensai_server.llm.embedder import Embedder
from sensai_server.llm.persona import DuplicatePersonaError, PersonaManager
from sensai_server.llm.provider import ProviderManager
from sensai_server.memory.persistent_db.base import get_db, set_db
from sensai_server.memory.persistent_db.error import DbConnectionError
from sensai_server.pipeline import ChatPipeline
from sensai_server.pipeline.modules import ModelStage, PersonaStage, RagStage
from sensai_server.pipeline.modules.guardrails import BadWordsGuard
from sensai_server.rag import Rag
from sensai_server.session import SessionRegistry
from sensai_server.tools import build_tools
from sensai_server.tools.tool_manager import ToolManager

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sensai_protocol import ClientMessage

    from sensai_server.config.settings import Settings
    from sensai_server.memory.persistent_db.adapter import PersistentDatabase
    from sensai_server.session.session import Session

logger = logging.getLogger(__name__)

# How long, in seconds, the startup probe waits for the Redis server of the semantic cache.
_REDIS_PROBE_TIMEOUT = 2

router = APIRouter()


def create_app(
    settings: Settings,
) -> FastAPI:
    """Build a new app serving the `/ws` route, logging clients in on the models of the configured providers.

    `settings` are the loaded settings, kept in the app state (see `app_state`). The persistent database is opened,
    and the default persona is stored in it if it is not there yet. The chat pipeline is built once
    (see `_build_pipeline`) and shared by every session: each `chat` prompt goes through it as the session's task. So
    is the tool manager (see `_build_tool_manager`): each session keeps only its own tool state. So is the ReAct loop
    the pipeline's model stage hands each exchange to: it keeps nothing between two exchanges. So is the semantic
    cache (see `_build_cache`), when there is one: its entries are cleared when the app starts and when it stops
    (see `_lifespan`).
    """
    pm: ProviderManager = ProviderManager(settings.llm)
    embedder = _build_embedder(settings, pm)
    tool_manager = _build_tool_manager(settings, embedder)
    react = ReAct(pm, tool_manager)
    set_db(settings.db.persistent)
    db = get_db()
    if db is None:
        msg = "the persistent database is not open"
        raise DbConnectionError(msg)
    persona_manager = PersonaManager(db)
    with contextlib.suppress(DuplicatePersonaError):
        persona_manager.create_default_persona()
    rag = _build_rag(settings, embedder, db)
    cache = _build_cache(settings, embedder)
    pipeline = _build_pipeline(react, rag)

    sessions = SessionRegistry(pm, persona_manager, tool_manager)
    commands = CommandRegistry()
    commands.set_commands(build_commands(sessions, commands, db, pm, tool_manager, rag=rag))

    app = FastAPI(lifespan=_lifespan)
    set_app_state(
        app,
        AppState(
            settings=settings,
            provider_manager=pm,
            embedder=embedder,
            tool_manager=tool_manager,
            react=react,
            db=db,
            persona_manager=persona_manager,
            registry=sessions,
            commands=commands,
            pipeline=pipeline,
            rag=rag,
            cache=cache,
        ),
    )
    app.include_router(router)
    return app


def _build_pipeline(react: ReAct, rag: Rag | None) -> ChatPipeline:
    """Build the chain every chat goes through: the guard, the persona, the RAG's extracts, then the ReAct loop.

    The guard comes first, so that a rejected prompt is never embedded. Without `rag`, there are no extracts.
    """
    pipeline = ChatPipeline(ModelStage(react)).use(BadWordsGuard()).use(PersonaStage())
    if rag is not None:
        pipeline.use(RagStage(rag))
    return pipeline


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


def _build_rag(settings: Settings, embedder: Embedder | None, db: PersistentDatabase) -> Rag | None:
    """Build the RAG shared by every session, on the Qdrant of the settings, or None without an embedder.

    Without it, the conversations have no RAG: at startup there is no client to tell, so it is logged.
    """
    if embedder is None:
        logger.warning("no embedder: the RAG is left out")
        return None
    location = settings.db.rag.location
    return Rag(location, embedder, db)


def _build_cache(settings: Settings, embedder: Embedder | None) -> SemanticCache | None:
    """Build the semantic cache shared by every session, on the Redis of the settings, or None without one.

    Without an embedder, or without a Redis server answering, the conversations have no cache: at startup there is no
    client to tell, so it is logged. `create_app` runs before the event loop does, so Redis is probed with a short-lived
    blocking client; the cache itself only uses its async client, from the event loop.
    """
    if embedder is None:
        logger.warning("no embedder: the semantic cache is left out")
        return None
    if not _redis_answers(settings.cache.redis_url):
        logger.warning("the Redis server does not answer: the semantic cache is left out")
        return None
    return SemanticCache(settings.cache.redis_url, embedder, settings.cache.similarity_threshold, settings.cache.ttl)


def _redis_answers(redis_url: str) -> bool:
    """Report whether the Redis server at `redis_url` answers a ping, within `_REDIS_PROBE_TIMEOUT` seconds."""
    client = SyncRedis.from_url(
        redis_url, socket_connect_timeout=_REDIS_PROBE_TIMEOUT, socket_timeout=_REDIS_PROBE_TIMEOUT
    )
    try:
        return bool(client.ping())
    except RedisError:
        return False
    finally:
        client.close()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Clear the semantic cache when the app starts, then close every session and clear the cache when it stops.

    At startup, the entries left are those of a previous run's sessions, which no longer exist: each login has a fresh
    client id. A failing clear is logged and never stops the app from starting or stopping.
    """
    state = app_state(app)
    if state.cache is not None:
        await _clear_cache(state.cache)
    try:
        yield
    finally:
        await state.registry.close_all()
        if state.cache is not None:
            await _clear_cache(state.cache)
            await state.cache.close()


async def _clear_cache(cache: SemanticCache) -> None:
    """Delete every entry of `cache`, logging a failure instead of raising it."""
    try:
        await cache.clear_all()
    except CacheError as error:
        logger.warning("the semantic cache could not be cleared: %s", error)


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
            await handlers.handle_chat(session, message, state.pipeline, state.db, state.rag)
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
