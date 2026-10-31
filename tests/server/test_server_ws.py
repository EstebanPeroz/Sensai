import threading
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect
from sensai_protocol import (
    AiChunk,
    AskUser,
    Cancel,
    Chat,
    ClientCommand,
    ClientMessage,
    Command,
    CommandContext,
    Complete,
    Completions,
    ConfirmAnswer,
    Confirmation,
    ConfirmTool,
    Error,
    Input,
    Login,
    Logout,
    RequestCommand,
    Role,
    ServerMessage,
    Session,
    System,
    UserAnswer,
    parse_server_message,
)
from sensai_server.app import _handlers as handlers
from sensai_server.app._state import app_state
from sensai_server.chat.react import ANSWER_INSTRUCTION, EFFORTS, ReAct
from sensai_server.commands import PersonaCommand
from sensai_server.commands._branch import Branch
from sensai_server.commands._effort import Effort as EffortCommand
from sensai_server.commands._help import Help
from sensai_server.commands._model import Model
from sensai_server.commands._tools import ToolsCommand
from sensai_server.pipeline import ChatPipeline
from sensai_server.pipeline.modules import ModelStage
from sensai_server.pipeline.modules.guardrails import WARNING
from sensai_server.session import SessionRegistry
from sensai_server.session.session import DEFAULT_EFFORT
from starlette.testclient import WebSocketTestSession

from tests.server.conftest import ASK, CONFIRM, FILE, MODEL, PERSONA_PATH, SLOW
from tests.server.fakes import FakeAdapter, chunk


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


class TestAppState:
    def test_model_stage_uses_the_shared_react(self, app: FastAPI) -> None:
        state = app_state(app)
        assert isinstance(state.react, ReAct)
        model = state.pipeline._model  # noqa: SLF001
        assert isinstance(model, ModelStage)
        assert model._react is state.react  # noqa: SLF001


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
            assert session.commands == ["/model", "/persona", "/branch", "/tools", "/effort", "/help"]
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


PERSONA_TOML = 'name = "Pirate"\ndescription = "Talks like a pirate"\nprompt = "Arr."\n'


def _request_file(ws: WebSocketTestSession) -> RequestCommand:
    """Start the `file` chat and return the `request_command` it sends."""
    _send(ws, Chat(prompt=FILE))
    assert _receive(ws) == Input(role=Role.USER, content=FILE)
    request = _receive(ws)
    assert isinstance(request, RequestCommand)
    assert (request.command, request.params) == (ClientCommand.READ_FILE, {"path": PERSONA_PATH})
    return request


class TestClientCommands:
    def test_file_content_is_parsed_by_the_server(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            request = _request_file(ws)
            _send(
                ws,
                CommandContext(request_id=request.request_id, command=request.command, ctx={"content": PERSONA_TOML}),
            )
            assert _receive(ws) == AiChunk(content="Pirate")
            assert _receive(ws) == AiChunk(done=True)

    @pytest.mark.parametrize(
        ("fields", "reason"),
        [
            ({"error": "file not found"}, "file not found"),
            ({"ctx": {"content": "name = "}}, "invalid TOML"),
            ({"ctx": {"content": 'name = "x"'}}, "missing key(s): description, prompt"),
        ],
        ids=["client-error", "invalid-toml", "missing-keys"],
    )
    def test_unusable_file_is_reported(self, client: TestClient, fields: dict, reason: str) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            request = _request_file(ws)
            _send(ws, CommandContext(request_id=request.request_id, command=request.command, **fields))
            answer = _receive(ws)
            assert isinstance(answer, AiChunk)
            assert answer.content is not None
            assert f"invalid persona file '{PERSONA_PATH}'" in answer.content
            assert reason in answer.content

    def test_answer_to_another_request_is_dropped(self, client: TestClient, registry: SessionRegistry) -> None:
        with client.websocket_connect("/ws") as ws:
            session = _login(ws)
            request = _request_file(ws)
            _send(ws, CommandContext(request_id=uuid4(), command=request.command, ctx={"content": PERSONA_TOML}))
            _sync(ws)
            server_session = registry.get(session.client_id)
            assert server_session is not None
            assert server_session.busy
            _send(ws, CommandContext(request_id=request.request_id, command=request.command, error="late"))
            assert "late" in (_receive(ws).content or "")  # type: ignore[union-attr]


CREATE = f"/persona create {PERSONA_PATH}"


def _run_command(ws: WebSocketTestSession, command: str) -> None:
    _send(ws, Command(command=command))
    assert _receive(ws) == Input(role=Role.USER, content=command, command=True)


def _request_persona_file(ws: WebSocketTestSession, path: str = PERSONA_PATH) -> RequestCommand:
    """Run `/persona create path` and return the `request_command` it sends."""
    _run_command(ws, f"/persona create {path}")
    request = _receive(ws)
    assert isinstance(request, RequestCommand)
    assert (request.command, request.params) == (ClientCommand.READ_FILE, {"path": path})
    return request


def _answer(ws: WebSocketTestSession, request: RequestCommand, **fields: object) -> None:
    _send(ws, CommandContext(request_id=request.request_id, command=request.command, **fields))  # type: ignore[arg-type]


def _system(ws: WebSocketTestSession) -> str:
    message = _receive(ws)
    assert isinstance(message, System)
    return message.message


def _stored_personas(ws: WebSocketTestSession) -> str:
    """The output of `/persona list`."""
    _run_command(ws, "/persona list")
    return _system(ws)


class TestPersonaCommand:
    def test_create_from_the_client_file(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _answer(ws, _request_persona_file(ws), ctx={"content": PERSONA_TOML})
            assert _system(ws) == f"Persona 'Pirate' created from '{PERSONA_PATH}'."
            assert "- Pirate: Talks like a pirate" in _stored_personas(ws)

    @pytest.mark.parametrize(
        ("fields", "reason"),
        [
            ({"error": "file not found"}, "file not found"),
            ({"ctx": {"content": "name = "}}, "invalid TOML"),
            ({"ctx": {"content": 'name = "Pirate"'}}, "missing key(s): description, prompt"),
        ],
        ids=["client-error", "invalid-toml", "missing-keys"],
    )
    def test_unusable_file_stores_nothing(self, client: TestClient, fields: dict, reason: str) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _answer(ws, _request_persona_file(ws), **fields)
            message = _system(ws)
            assert f"invalid persona file '{PERSONA_PATH}'" in message
            assert reason in message
            assert "Pirate" not in _stored_personas(ws)

    def test_answer_to_another_request_is_dropped(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            request = _request_persona_file(ws)
            _send(ws, CommandContext(request_id=uuid4(), command=request.command, ctx={"content": PERSONA_TOML}))
            _sync(ws)
            _answer(ws, request, ctx={"content": PERSONA_TOML})
            assert _system(ws) == f"Persona 'Pirate' created from '{PERSONA_PATH}'."

    def test_chat_stops_the_waiting_command(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            request = _request_persona_file(ws)
            _send(ws, Chat(prompt="hi"))
            assert _system(ws) == f"Command '{CREATE}' stopped: a new message was sent."
            assert _receive(ws) == Input(role=Role.USER, content="hi")
            assert _receive(ws) == AiChunk(content="hi")
            assert _receive(ws) == AiChunk(done=True)
            _answer(ws, request, ctx={"content": PERSONA_TOML})
            _sync(ws)
            assert "Pirate" not in _stored_personas(ws)

    def test_command_stops_the_waiting_command(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            first = _request_persona_file(ws)
            _send(ws, Command(command="/persona create y.toml"))
            assert _system(ws) == f"Command '{CREATE}' stopped: a new message was sent."
            assert _receive(ws) == Input(role=Role.USER, content="/persona create y.toml", command=True)
            second = _receive(ws)
            assert isinstance(second, RequestCommand)
            assert second.params == {"path": "y.toml"}
            _answer(ws, first, ctx={"content": PERSONA_TOML})
            _sync(ws)
            _answer(ws, second, ctx={"content": PERSONA_TOML.replace("Pirate", "Ninja")})
            assert _system(ws) == "Persona 'Ninja' created from 'y.toml'."
            stored = _stored_personas(ws)
            assert "Ninja" in stored
            assert "Pirate" not in stored

    def test_create_while_a_chat_runs_is_refused(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, Chat(prompt=SLOW))
            assert _receive(ws) == Input(role=Role.USER, content=SLOW)
            assert _receive(ws) == AiChunk(content=SLOW)
            _send(ws, Command(command=CREATE))
            assert "already running" in _receive_error(ws).message
            _send(ws, Cancel())
            _sync(ws)

    def test_chat_while_a_tool_waits_is_refused(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, Chat(prompt=CONFIRM))
            _receive(ws)
            assert isinstance(_receive(ws), ConfirmTool)
            _send(ws, Chat(prompt="hi"))
            assert "already running" in _receive_error(ws).message

    def test_cancel_while_waiting_stores_nothing(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            request = _request_persona_file(ws)
            _send(ws, Cancel())
            _sync(ws)
            _answer(ws, request, ctx={"content": PERSONA_TOML})
            _sync(ws)
            assert "Pirate" not in _stored_personas(ws)

    def test_complete_persona_names(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, Complete(text="/persona load "))
            assert _receive(ws) == Completions(text="/persona load ", values=["Analyst"])


class TestCommandAndComplete:
    def test_command(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, Command(command="/help"))
            assert _receive(ws) == Input(role=Role.USER, content="/help", command=True)
            assert _receive(ws) == System(
                message=(
                    f"Available commands:\n- branch: {Branch.help}\n- effort: {EffortCommand.help}\n"
                    f"- help: {Help.help}\n- model: {Model.help}\n- persona: {PersonaCommand.help}\n"
                    f"- tools: {ToolsCommand.help}"
                )
            )

    def test_complete(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, Complete(text="/model "))
            assert _receive(ws) == Completions(text="/model ", values=[MODEL, "b"])


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
            (
                "handle_command_context",
                CommandContext(request_id=uuid4(), command=ClientCommand.READ_FILE, error="x"),
            ),
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

    def test_chat_reaches_its_handler_with_the_pipeline_and_db(
        self, app: FastAPI, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handler = AsyncMock()
        monkeypatch.setattr(handlers, "handle_chat", handler)
        with client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, Chat(prompt="hi"))
            _sync(ws)
        assert handler.await_args is not None
        _, _, pipeline, db = handler.await_args.args
        assert pipeline is app_state(app).pipeline
        assert db is app_state(app).db

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


class TestEndToEnd:
    """The real chat, through the pipeline built by `create_app`, answered by `FakeAdapter`."""

    @pytest.fixture(autouse=True)
    def replies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(FakeAdapter, "replies", [chunk("Hel"), chunk("lo", done=True)])
        # `FakeAdapter` answers every call with the same stream: the default effort makes a single answering call.
        monkeypatch.setitem(EFFORTS, DEFAULT_EFFORT, 0)

    def test_app_state_has_a_pipeline(self, real_app: FastAPI) -> None:
        assert isinstance(app_state(real_app).pipeline, ChatPipeline)

    def test_chat_streams_the_model_answer(self, real_client: TestClient) -> None:
        with real_client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, Chat(prompt="hello"))
            assert _receive(ws) == Input(role=Role.USER, content="hello")
            assert _receive(ws) == AiChunk(content="Hel", done=False)
            assert _receive(ws) == AiChunk(content="lo", done=True)
            _sync(ws)  # the prompt is echoed only once, and nothing follows the last chunk
        assert FakeAdapter.last_payload is not None
        assert FakeAdapter.last_payload["model"] == MODEL
        assert FakeAdapter.last_payload["messages"][-2:] == [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": ANSWER_INSTRUCTION},
        ]

    def test_bad_word_is_answered_without_the_model(self, real_client: TestClient) -> None:
        with real_client.websocket_connect("/ws") as ws:
            _login(ws)
            _send(ws, Chat(prompt="fuck"))
            assert _receive(ws) == Input(role=Role.USER, content="fuck")
            assert _receive(ws) == AiChunk(content=WARNING, done=True)
            _sync(ws)
        assert FakeAdapter.last_payload is None

    def test_cancel_mid_stream_then_chat(self, real_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        block = threading.Event()
        monkeypatch.setattr(FakeAdapter, "block", block)
        try:
            with real_client.websocket_connect("/ws") as ws:
                _login(ws)
                _send(ws, Chat(prompt="hello"))
                _receive(ws)
                assert _receive(ws) == AiChunk(content="Hel", done=False)
                _send(ws, Cancel())
                _send(ws, Chat(prompt="again"))
                assert _receive(ws) == Input(role=Role.USER, content="again")
                block.set()
                assert FakeAdapter.closed.wait(timeout=5)
                assert _receive(ws) == AiChunk(content="Hel", done=False)
                assert _receive(ws) == AiChunk(content="lo", done=True)
        finally:
            block.set()
        assert FakeAdapter.last_payload is not None
        # The cancelled answer was never saved: the second payload holds both prompts and no answer.
        assert [message["role"] for message in FakeAdapter.last_payload["messages"]][-2:] == ["user", "user"]
        assert all(message["role"] != "assistant" for message in FakeAdapter.last_payload["messages"])
