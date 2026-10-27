from __future__ import annotations

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from sensai_protocol import Cancel, Chat, Command, Complete, Login, Logout, ProtocolError, parse_client_message

from sensai_server import handlers
from sensai_server.outbound import Outbound

router = APIRouter()


def create_app() -> FastAPI:
    """Build a new app serving the `/ws` route."""
    app = FastAPI()
    app.include_router(router)
    return app


@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    """Receive the client's messages and dispatch each one to its handler, until logout or disconnection.

    A frame that is not a valid client message is answered with an `error`, and the connection stays open.
    """
    await ws.accept()
    out = Outbound(ws)
    try:
        async for raw in ws.iter_text():
            try:
                message = parse_client_message(raw)
            except ProtocolError as err:
                await out.send_error(str(err))
                continue
            match message:
                case Login():
                    await handlers.handle_login(out, message)
                case Logout():
                    await handlers.handle_logout(out, message)
                    await ws.close()
                    break
                case Chat():
                    await handlers.handle_chat(out, message)
                case Command():
                    await handlers.handle_command(out, message)
                case Cancel():
                    await handlers.handle_cancel(out, message)
                case Complete():
                    await handlers.handle_complete(out, message)
    except WebSocketDisconnect:
        pass
    # milestone 5: free the session here, in a `finally`, whatever ended the connection
