from __future__ import annotations

import contextlib
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

from sensai_server import handlers
from sensai_server.llm.persona import DuplicatePersonaError, PersonaManager
from sensai_server.llm.provider import ProviderManager
from sensai_server.memory.persistent_db.base import get_db, set_db
from sensai_server.memory.persistent_db.error import DbConnectionError
from sensai_server.outbound import Outbound
from sensai_server.session import SessionRegistry

if TYPE_CHECKING:
    from sensai_protocol import ClientMessage

    from sensai_server.config.settings import Settings
    from sensai_server.handlers import ChatRunner, CommandRunner
    from sensai_server.session.session import Session

router = APIRouter()


def create_app(
    settings: Settings,
    *,
    chat: ChatRunner = handlers.placeholder_chat,
    command: CommandRunner = handlers.placeholder_command,
) -> FastAPI:
    """Build a new app serving the `/ws` route, logging clients in on the models of the configured providers.

    `settings` are the loaded settings, kept in the app state. The persistent database is opened, and the default
    persona is stored in it if it is not there yet. `chat` and `command` run a chat prompt and a command as a
    session's task.
    """
    pm: ProviderManager = ProviderManager(settings.llm)
    set_db(settings.db.persistent)
    db = get_db()
    if db is None:
        msg = "the persistent database is not open"
        raise DbConnectionError(msg)
    persona_manager = PersonaManager(db)
    with contextlib.suppress(DuplicatePersonaError):
        persona_manager.create_default_persona()

    app = FastAPI()
    app.state.settings = settings
    app.state.provider_manager = pm
    app.state.db = db
    app.state.persona_manager = persona_manager
    app.state.registry = SessionRegistry(pm, persona_manager)
    app.state.chat = chat
    app.state.command = command
    app.include_router(router)
    return app


@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    """Receive the client's messages and dispatch each one to its handler, until logout or disconnection.

    A frame that is not a valid client message, or any message but `login` before a session is open, is answered
    with an `error`, and the connection stays open. However the connection ends, its session is freed.
    """
    await ws.accept()
    registry: SessionRegistry = ws.app.state.registry
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
                session = await handlers.handle_login(registry, out, session, message)
                continue
            if session is None:
                await out.send_error("Login first.")
                continue
            if not await _dispatch(ws, session, message):
                break
    except WebSocketDisconnect:
        pass
    finally:
        if session is not None:
            await registry.close(session.client_id)


async def _dispatch(ws: WebSocket, session: Session, message: ClientMessage) -> bool:
    """Dispatch a message, other than `login`, of a logged-in client to its handler.

    Returns False once the connection must end.
    """
    state = ws.app.state
    match message:
        case Logout():
            await handlers.handle_logout(state.registry, session, message)
            await ws.close()
            return False
        case Chat():
            await handlers.handle_chat(session, message, state.chat)
        case Command():
            await handlers.handle_command(session, message, state.command)
        case Cancel():
            await handlers.handle_cancel(session, message)
        case Complete():
            await handlers.handle_complete(session, message)
        case ConfirmAnswer():
            await handlers.handle_confirm_answer(session, message)
        case UserAnswer():
            await handlers.handle_user_answer(session, message)
        case CommandContext():
            await handlers.handle_command_context(session, message)
    return True
