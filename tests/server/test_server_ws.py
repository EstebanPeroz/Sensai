from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect
from sensai_protocol import (
    AiChunk,
    AskUser,
    Cancel,
    Chat,
    ClientMessage,
    Command,
    Complete,
    Completions,
    ConfirmAnswer,
    Confirmation,
    ConfirmTool,
    Error,
    Input,
    Login,
    Logout,
    Role,
    ServerMessage,
    Session,
    System,
    UserAnswer,
    parse_server_message,
)
from sensai_server import handlers
from sensai_server.session import SessionRegistry
from starlette.testclient import WebSocketTestSession

from tests.server.conftest import ASK, CONFIRM, MODEL, SLOW


def _receive(ws: WebSocketTestSession) -> ServerMessage:
    return parse_server_message(ws.receive_text())


def _receive_error(ws: WebSocketTestSession) -> Error:
    message = _receive(ws)
    assert isinstance(message, Error)
    return message


def _send(ws: WebSocketTestSession, message: ClientMessage) -> None:
    ws.send_text(message.model_dump_json())


def _sync(ws: WebSocketTestSession) -> None:
    """Wait until the frames sent before are handled: the server answers this invalid one in order."""
    ws.send_text("{}")
    _receive_error(ws)


def _login(ws: WebSocketTestSession) -> Session:
    _send(ws, Login(model=MODEL))
    message = _receive(ws)
    assert isinstance(message, Session)
    return message


class TestProtocolErrors:
    def test_invalid_json_keeps_connection(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_text("not json")
            _receive_error(ws)
            _login(ws)

    def test_unknown_type(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "nope"})
            _receive_error(ws)


class TestLogin:
    @pytest.mark.parametrize(
        "message", [Chat(prompt="hi"), Command(command="/help"), Cancel(), Complete(text="/m"), Logout()]
    )
    def test_message_before_login(self, client: TestClient, registry: SessionRegistry, message: ClientMessage) -> None:
        with client.websocket_connect("/ws") as ws:
            _send(ws, message)
            assert "login" in _receive_error(ws).message.lower()
            _login(ws)
            assert len(registry) == 1

    def test_login(self, client: TestClient, registry: SessionRegistry) -> None:
        with client.websocket_connect("/ws") as ws:
            session = _login(ws)
            assert session.model == MODEL
            assert session.commands == []
            assert registry.get(session.client_id) is not None

    def test_login_twice_keeps_session(self, client: TestClient, registry: SessionRegistry) -> None:
        with client.websocket_connect("/ws") as ws:
            session = _login(ws)
            _send(ws, Login(model=MODEL))
            _receive_error(ws)
            assert len(registry) == 1
            assert registry.get(session.client_id) is not None

    def test_unknown_model_can_retry(self, client: TestClient, registry: SessionRegistry) -> None:
        with client.websocket_connect("/ws") as ws:
            _send(ws, Login(model="nope"))
            assert f"Available models: {MODEL}, b" in _receive_error(ws).message
            assert len(registry) == 0
            _login(ws)
            assert len(registry) == 1


class TestChat:
    def test_chat_streams(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, Chat(prompt="hi"))
            assert _receive(ws) == Input(role=Role.USER, content="hi")
            assert _receive(ws) == AiChunk(content="hi")
            assert _receive(ws) == AiChunk(done=True)

    def test_chat_while_busy(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, Chat(prompt=SLOW))
            assert _receive(ws) == Input(role=Role.USER, content=SLOW)
            assert _receive(ws) == AiChunk(content=SLOW)
            _send(ws, Chat(prompt="hi"))
            _receive_error(ws)
            _send(ws, Command(command="/help"))
            _receive_error(ws)

    def test_cancel_stops_stream(self, client: TestClient, registry: SessionRegistry) -> None:
        with client.websocket_connect("/ws") as ws:
            session = _login(ws)
            _send(ws, Chat(prompt=SLOW))
            _receive(ws)
            _receive(ws)
            _send(ws, Cancel())
            _sync(ws)  # nothing was sent for the cancelled answer
            server_session = registry.get(session.client_id)
            assert server_session is not None
            assert not server_session.busy
            _send(ws, Chat(prompt="hi"))
            assert _receive(ws) == Input(role=Role.USER, content="hi")

    def test_cancel_then_chat_right_away(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, Chat(prompt=SLOW))
            _receive(ws)
            _receive(ws)
            _send(ws, Cancel())
            _send(ws, Chat(prompt="hi"))
            assert _receive(ws) == Input(role=Role.USER, content="hi")

    def test_cancel_when_idle(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, Cancel())
            _sync(ws)


class TestQuestions:
    @pytest.mark.parametrize("answer", list(Confirmation))
    def test_confirm_answer_resumes_the_task(self, client: TestClient, answer: Confirmation) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, Chat(prompt=CONFIRM))
            assert _receive(ws) == Input(role=Role.USER, content=CONFIRM)
            request = _receive(ws)
            assert isinstance(request, ConfirmTool)
            assert (request.tool_name, request.arguments) == ("web_search", {"query": "paris"})
            _send(ws, ConfirmAnswer(request_id=request.request_id, answer=answer))
            assert _receive(ws) == AiChunk(content=answer.value)
            assert _receive(ws) == AiChunk(done=True)

    def test_user_answer_resumes_the_task_and_is_shown(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, Chat(prompt=ASK))
            _receive(ws)
            request = _receive(ws)
            assert request == AskUser(request_id=request.request_id, question="Which city?")  # type: ignore[union-attr]
            _send(ws, UserAnswer(request_id=request.request_id, answer="Paris"))  # type: ignore[union-attr]
            assert _receive(ws) == System(message="Sensai asked: Which city?\n> Paris")
            assert _receive(ws) == AiChunk(content="Paris")

    def test_stale_answer_is_ignored(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, UserAnswer(request_id=uuid4(), answer="late"))
            _sync(ws)

    def test_cancel_while_asking_frees_the_session(self, client: TestClient, registry: SessionRegistry) -> None:
        with client.websocket_connect("/ws") as ws:
            session = _login(ws)
            _send(ws, Chat(prompt=CONFIRM))
            _receive(ws)
            request = _receive(ws)
            _send(ws, Cancel())
            _sync(ws)
            server_session = registry.get(session.client_id)
            assert server_session is not None
            assert not server_session.busy
            _send(ws, ConfirmAnswer(request_id=request.request_id, answer=Confirmation.ALLOW))  # type: ignore[union-attr]
            _sync(ws)


class TestCommandAndComplete:
    def test_command(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, Command(command="/help"))
            assert _receive(ws) == Input(role=Role.USER, content="/help", command=True)
            assert _receive(ws) == System(message="ran /help")

    def test_complete(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, Complete(text="/model "))
            assert _receive(ws) == Completions(text="/model ", values=[])


class TestDispatch:
    @pytest.mark.parametrize(
        ("name", "message"),
        [
            ("handle_chat", Chat(prompt="hi")),
            ("handle_command", Command(command="/model x")),
            ("handle_cancel", Cancel()),
            ("handle_complete", Complete(text="/model ")),
            ("handle_confirm_answer", ConfirmAnswer(request_id=uuid4(), answer=Confirmation.ALLOW)),
            ("handle_user_answer", UserAnswer(request_id=uuid4(), answer="x")),
        ],
    )
    def test_message_reaches_its_handler(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, name: str, message: ClientMessage
    ) -> None:
        handler = AsyncMock()
        monkeypatch.setattr(handlers, name, handler)
        with client.websocket_connect("/ws") as ws:
            session = _login(ws)
            _send(ws, message)
            _sync(ws)
        handler.assert_awaited_once()
        assert handler.await_args is not None
        received_session, received_message = handler.await_args.args[:2]
        assert received_session.client_id == session.client_id
        assert received_message == message

    def test_logout_reaches_its_handler_with_the_registry(
        self, client: TestClient, registry: SessionRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handler = AsyncMock()
        monkeypatch.setattr(handlers, "handle_logout", handler)
        with client.websocket_connect("/ws") as ws:
            session = _login(ws)
            _send(ws, Logout())
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()
        handler.assert_awaited_once()
        assert handler.await_args is not None
        received_registry, received_session, received_message = handler.await_args.args
        assert received_registry is registry
        assert received_session.client_id == session.client_id
        assert received_message == Logout()


class TestCleanup:
    def test_logout_closes_socket_and_frees_session(self, client: TestClient, registry: SessionRegistry) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, Chat(prompt=SLOW))
            _receive(ws)
            _send(ws, Logout())
            _receive(ws)
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()
        assert len(registry) == 0

    def test_dropped_connection_frees_session(self, client: TestClient, registry: SessionRegistry) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, Chat(prompt=SLOW))
            _receive(ws)
            _receive(ws)
            assert len(registry) == 1
        assert len(registry) == 0

    def test_two_connections_are_isolated(self, client: TestClient, registry: SessionRegistry) -> None:
        with client.websocket_connect("/ws") as first, client.websocket_connect("/ws") as second:
            first_session = _login(first)
            second_session = _login(second)
            assert first_session.client_id != second_session.client_id
            assert len(registry) == 2
            _send(first, Chat(prompt="hi"))
            assert _receive(first) == Input(role=Role.USER, content="hi")
            _sync(second)  # the next frame second receives is its own answer, not first's chat
            _send(second, Complete(text="/x"))
            assert _receive(second) == Completions(text="/x", values=[])
        assert len(registry) == 0
