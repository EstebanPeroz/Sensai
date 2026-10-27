from unittest.mock import ANY, AsyncMock

import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect
from sensai_protocol import (
    Cancel,
    Chat,
    ClientMessage,
    Command,
    Complete,
    Error,
    Login,
    Logout,
    parse_server_message,
)
from sensai_server import handlers
from starlette.testclient import WebSocketTestSession


def _receive_error(ws: WebSocketTestSession) -> Error:
    message = parse_server_message(ws.receive_text())
    assert isinstance(message, Error)
    return message


def _sync(ws: WebSocketTestSession) -> None:
    """Wait until the frames sent before are handled: the server answers this invalid one in order."""
    ws.send_text("{}")
    _receive_error(ws)


class TestProtocolErrors:
    def test_invalid_json_keeps_connection(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        handle_chat = AsyncMock()
        monkeypatch.setattr(handlers, "handle_chat", handle_chat)
        with client.websocket_connect("/ws") as ws:
            ws.send_text("not json")
            _receive_error(ws)
            ws.send_text(Chat(prompt="hi").model_dump_json())
            _sync(ws)
        handle_chat.assert_awaited_once_with(ANY, Chat(prompt="hi"))

    def test_unknown_type(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "nope"})
            _receive_error(ws)


class TestDispatch:
    @pytest.mark.parametrize(
        ("name", "message"),
        [
            ("handle_login", Login(model="fake-model")),
            ("handle_chat", Chat(prompt="hi")),
            ("handle_command", Command(command="/model x")),
            ("handle_cancel", Cancel()),
            ("handle_complete", Complete(text="/model ")),
        ],
    )
    def test_message_reaches_its_handler(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, name: str, message: ClientMessage
    ) -> None:
        handler = AsyncMock()
        monkeypatch.setattr(handlers, name, handler)
        with client.websocket_connect("/ws") as ws:
            ws.send_text(message.model_dump_json())
            _sync(ws)
        handler.assert_awaited_once_with(ANY, message)

    def test_logout_closes_socket(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        handle_logout = AsyncMock()
        monkeypatch.setattr(handlers, "handle_logout", handle_logout)
        with client.websocket_connect("/ws") as ws:
            ws.send_text(Logout().model_dump_json())
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()
        handle_logout.assert_awaited_once_with(ANY, Logout())
