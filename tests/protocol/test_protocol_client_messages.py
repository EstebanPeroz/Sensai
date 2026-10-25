import json
import pkgutil
import re
from pathlib import Path
from uuid import uuid4

import pydantic
import pytest
import sensai_protocol
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
    Input,
    Login,
    Logout,
    ProtocolError,
    Role,
    ServerMessage,
    Session,
    System,
    parse_client_message,
)

CLIENT_MESSAGES: list[ClientMessage] = [
    Login(model="qwen3:1.7b"),
    Logout(),
    Chat(prompt="Explain the ReAct loop"),
    Command(command="/model llama3:8b"),
    Cancel(),
    Complete(text="/persona load an"),
]

SERVER_MESSAGES: list[ServerMessage] = [
    Session(client_id=uuid4(), model="qwen3:1.7b", commands=["/help"], effort="medium"),
    AiChunk(content="hi", done=True),
    Input(role=Role.USER, content="hi"),
    System(message="ok"),
    Error(message="boom"),
    History(messages=[]),
    Effort(level="high"),
    Completions(text="/model ", values=["qwen3:1.7b"]),
]


class TestClientMessages:
    @pytest.mark.parametrize("message", CLIENT_MESSAGES, ids=lambda m: m.type)
    def test_round_trip(self, message: ClientMessage) -> None:
        parsed = parse_client_message(message.model_dump_json())

        assert parsed == message
        assert type(parsed) is type(message)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ({"type": "login", "model": "qwen3:1.7b"}, Login),
            ({"type": "logout"}, Logout),
            ({"type": "chat", "prompt": "hi"}, Chat),
            ({"type": "command", "command": "/help"}, Command),
            ({"type": "cancel"}, Cancel),
            ({"type": "complete", "text": "/mo"}, Complete),
        ],
    )
    def test_type_parses_into_its_class(self, raw: dict, expected: type) -> None:
        assert type(parse_client_message(json.dumps(raw))) is expected

    def test_parses_bytes(self) -> None:
        assert parse_client_message(b'{"type": "cancel"}') == Cancel()

    @pytest.mark.parametrize(
        "raw",
        [
            '{"type": "unknown"}',
            '{"model": "qwen3:1.7b"}',
            '{"type": "login"}',
            '{"type": "login", "model": "qwen3:1.7b", "extra": 1}',
            '{"type": "cancel", "modle": "typo"}',
            '{"type": "chat", "prompt": 1}',
            "not json",
            "",
            "[]",
        ],
        ids=[
            "unknown-type",
            "missing-type",
            "missing-field",
            "extra-field",
            "typo",
            "wrong-type",
            "invalid-json",
            "empty",
            "not-an-object",
        ],
    )
    def test_invalid_frame_raises_protocol_error(self, raw: str) -> None:
        with pytest.raises(ProtocolError):
            parse_client_message(raw)

    @pytest.mark.parametrize("message", SERVER_MESSAGES, ids=lambda m: m.type)
    def test_server_message_is_rejected(self, message: ServerMessage) -> None:
        with pytest.raises(ProtocolError):
            parse_client_message(message.model_dump_json())

    def test_is_immutable(self) -> None:
        message = Chat(prompt="hi")

        with pytest.raises(pydantic.ValidationError):
            message.prompt = "changed"  # type: ignore[misc]

    def test_type_is_serialized(self) -> None:
        assert json.loads(Login(model="m").model_dump_json()) == {"type": "login", "model": "m"}


class TestProtocolPackage:
    def test_protocol_error_hides_pydantic(self) -> None:
        assert not issubclass(ProtocolError, pydantic.ValidationError)

    def test_imports_nothing_from_sensai(self) -> None:
        package_dir = Path(sensai_protocol.__file__).parent
        modules = [package_dir / f"{info.name}.py" for info in pkgutil.iter_modules([str(package_dir)])]
        forbidden = re.compile(r"^\s*(from|import)\s+(sensai|sensai_server|sensai_tui)\b", re.MULTILINE)

        assert modules
        for module in [*modules, package_dir / "__init__.py"]:
            assert not forbidden.search(module.read_text()), module.name
