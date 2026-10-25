import json
from uuid import uuid4

import pytest
from sensai_protocol import (
    AiChunk,
    Cancel,
    Chat,
    ClientMessage,
    Command,
    Complete,
    Completions,
    Effort,
    Error,
    History,
    HistoryMessage,
    Input,
    Login,
    Logout,
    ProtocolError,
    Role,
    ServerMessage,
    Session,
    System,
    parse_server_message,
)

SERVER_MESSAGES: list[ServerMessage] = [
    Session(client_id=uuid4(), model="qwen3:1.7b", commands=["/help", "/model"], effort="medium"),
    Session(client_id=uuid4(), model="qwen3:1.7b", commands=[], effort="off"),
    AiChunk(content="Hello", thinking="Let me think"),
    AiChunk(tool_calls=[{"type": "function", "function": {"name": "echo", "arguments": {"city": "Paris"}}}]),
    AiChunk(done=True),
    AiChunk(done=True, error="model not found"),
    Input(role=Role.USER, content="hi"),
    Input(role=Role.USER, content="/help", command=True),
    System(message="Model loaded"),
    System(message=" more", append=True),
    Error(message="boom"),
    History(messages=[]),
    History(
        messages=[
            HistoryMessage(uuid=None, role=Role.SYSTEM, content="You are helpful"),
            HistoryMessage(uuid=uuid4(), role=Role.USER, content="hi"),
            HistoryMessage(uuid=uuid4(), role=Role.ASSISTANT, content="hello"),
        ]
    ),
    Effort(level="high"),
    Completions(text="/model ", values=["qwen3:1.7b", "llama3:8b"]),
    Completions(text="/persona load x", values=[]),
]

CLIENT_MESSAGES: list[ClientMessage] = [
    Login(model="qwen3:1.7b"),
    Logout(),
    Chat(prompt="hi"),
    Command(command="/help"),
    Cancel(),
    Complete(text="/mo"),
]

CLIENT_ID = "6f1c1e2a-4b0e-4d8e-9a3c-2f9b1d7e5a10"


class TestServerMessages:
    @pytest.mark.parametrize("message", SERVER_MESSAGES, ids=lambda m: m.type)
    def test_round_trip(self, message: ServerMessage) -> None:
        parsed = parse_server_message(message.model_dump_json())

        assert parsed == message
        assert type(parsed) is type(message)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ({"type": "session", "client_id": CLIENT_ID, "model": "m", "commands": [], "effort": "low"}, Session),
            ({"type": "ai_chunk", "content": "hi"}, AiChunk),
            ({"type": "input", "role": "user", "content": "hi"}, Input),
            ({"type": "system", "message": "ok"}, System),
            ({"type": "error", "message": "boom"}, Error),
            ({"type": "history", "messages": [{"uuid": None, "role": "user", "content": "hi"}]}, History),
            ({"type": "effort", "level": "high"}, Effort),
            ({"type": "completions", "text": "/mo", "values": ["/model"]}, Completions),
        ],
    )
    def test_type_parses_into_its_class(self, raw: dict, expected: type) -> None:
        assert type(parse_server_message(json.dumps(raw))) is expected

    def test_defaults(self) -> None:
        chunk = parse_server_message('{"type": "ai_chunk"}')

        assert chunk == AiChunk(content=None, thinking=None, tool_calls=None, done=False, error=None)
        assert parse_server_message('{"type": "input", "role": "user", "content": "x"}') == Input(
            role=Role.USER, content="x", command=False
        )
        assert parse_server_message('{"type": "system", "message": "x"}') == System(message="x", append=False)

    def test_session_client_id_is_a_uuid(self) -> None:
        raw = {"type": "session", "client_id": CLIENT_ID, "model": "m", "commands": [], "effort": "low"}

        message = parse_server_message(json.dumps(raw))

        assert isinstance(message, Session)
        assert str(message.client_id) == CLIENT_ID

    def test_roles_match_chat_endpoint(self) -> None:
        assert [role.value for role in Role] == ["system", "user", "assistant", "tool"]

    @pytest.mark.parametrize(
        "raw",
        [
            '{"type": "unknown"}',
            '{"message": "boom"}',
            '{"type": "error"}',
            '{"type": "error", "message": "boom", "extra": 1}',
            '{"type": "history", "messages": [{"uuid": null, "role": "user", "content": "hi", "extra": 1}]}',
            '{"type": "input", "role": "robot", "content": "hi"}',
            '{"type": "session", "client_id": "not-a-uuid", "model": "m", "commands": [], "effort": "low"}',
            "not json",
        ],
        ids=[
            "unknown-type",
            "missing-type",
            "missing-field",
            "extra-field",
            "nested-extra-field",
            "invalid-role",
            "invalid-uuid",
            "invalid-json",
        ],
    )
    def test_invalid_frame_raises_protocol_error(self, raw: str) -> None:
        with pytest.raises(ProtocolError):
            parse_server_message(raw)

    @pytest.mark.parametrize("message", CLIENT_MESSAGES, ids=lambda m: m.type)
    def test_client_message_is_rejected(self, message: ClientMessage) -> None:
        with pytest.raises(ProtocolError):
            parse_server_message(message.model_dump_json())
