from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from sensai_protocol import (
    Cancel,
    Chat,
    Command,
    Complete,
    ConfirmAnswer,
    Login,
    Logout,
    ProtocolError,
    UserAnswer,
    parse_client_message,
)

from sensai_server import handlers
from sensai_server.models import StaticModelCatalog
from sensai_server.outbound import Outbound
from sensai_server.session import SessionRegistry

if TYPE_CHECKING:
    from sensai_protocol import ClientMessage

    from sensai_server.handlers import ChatRunner, CommandRunner
    from sensai_server.session.session import Session

router = APIRouter()


def create_app(
    *,
    chat: ChatRunner = handlers.placeholder_chat,
    command: CommandRunner = handlers.placeholder_command,
) -> FastAPI:
    """Build a new app serving the `/ws` route, logging clients in on the models of `catalog`.

    `chat` and `command` run a chat prompt and a command as a session's task.
    """
    app = FastAPI()
    app.state.registry = SessionRegistry(StaticModelCatalog())
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
            await handlers.handle_logout(session, message)
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
    return True
