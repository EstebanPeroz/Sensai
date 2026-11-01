import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sensai_protocol import (
    AiChunk,
    AskCompression,
    AskUser,
    Cancel,
    Chat,
    ClientCommand,
    ClientMessage,
    Command,
    CommandContext,
    Complete,
    Completions,
    CompressionAnswer,
    CompressionChoice,
    ConfirmAnswer,
    Confirmation,
    ConfirmTool,
    ContextUsage,
    Dashboard,
    DashboardDrift,
    DashboardError,
    DashboardGroup,
    DashboardReport,
    DashboardSummary,
    Effort,
    Error,
    History,
    HistoryMessage,
    Input,
    Login,
    Logout,
    Period,
    ProtocolError,
    RequestCommand,
    Role,
    ServerMessage,
    Session,
    System,
    UserAnswer,
    parse_server_message,
)

SUMMARY = DashboardSummary(
    calls=4,
    errors=1,
    interrupted=1,
    latency_p50_ms=900.0,
    latency_p95_ms=None,
    first_token_p50_ms=120.0,
    tokens_per_s=None,
    prompt_tokens=10,
    completion_tokens=20,
)
REPORT = DashboardReport(
    period=Period.DAY,
    start=datetime(2026, 10, 13, 12, 0, tzinfo=UTC),
    end=datetime(2026, 10, 14, 12, 0, tzinfo=UTC),
    summary=SUMMARY,
    calls_series=[0, 3, 1],
    latency_series=[None, 800.0, 1000.0],
    groups=[DashboardGroup(model="qwen", persona=None, summary=SUMMARY)],
    drifts=[DashboardDrift(model="qwen", persona="analyst", baseline_p50_ms=1000.0, recent_p50_ms=1500.0)],
    recent_errors=[DashboardError(created_at=datetime(2026, 10, 14, 11, 0, tzinfo=UTC), model="qwen", error="boom")],
    baseline_periods=7,
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
    ConfirmTool(request_id=uuid4(), tool_name="web_search", arguments={"query": "paris", "limit": 3}),
    ConfirmTool(request_id=uuid4(), tool_name="now", arguments={}),
    AskUser(request_id=uuid4(), question="Which city?"),
    Dashboard(report=REPORT),
    RequestCommand(request_id=uuid4(), command=ClientCommand.READ_FILE, params={"path": "./me.toml"}),
    ContextUsage(used=2048, budget=8192, warning_threshold=0.8),
    ContextUsage(used=None, budget=None, warning_threshold=0.8),
    AskCompression(request_id=uuid4(), used=900, budget=1000, warning_threshold=0.8),
]

CLIENT_MESSAGES: list[ClientMessage] = [
    Login(model="qwen3:1.7b"),
    Logout(),
    Chat(prompt="hi"),
    Command(command="/help"),
    Cancel(),
    Complete(text="/mo"),
    ConfirmAnswer(request_id=uuid4(), answer=Confirmation.ALWAYS),
    UserAnswer(request_id=uuid4(), answer=None),
    CommandContext(request_id=uuid4(), command=ClientCommand.READ_FILE, ctx={"content": "x"}),
    CompressionAnswer(request_id=uuid4(), answer=CompressionChoice.NEW_CONVERSATION),
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
            ({"type": "confirm_tool", "request_id": CLIENT_ID, "tool_name": "t", "arguments": {}}, ConfirmTool),
            ({"type": "ask_user", "request_id": CLIENT_ID, "question": "why?"}, AskUser),
            (
                {"type": "request_command", "request_id": CLIENT_ID, "command": "read_file", "params": {"path": "x"}},
                RequestCommand,
            ),
            ({"type": "context_usage", "used": None, "budget": 8192, "warning_threshold": 0.8}, ContextUsage),
            (
                {"type": "ask_compression", "request_id": CLIENT_ID, "used": 1, "budget": 2, "warning_threshold": 0.8},
                AskCompression,
            ),
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

    def test_dashboard_derived_figures(self) -> None:
        message = parse_server_message(Dashboard(report=REPORT).model_dump_json())

        assert isinstance(message, Dashboard)
        assert message.report.summary.error_rate == 0.25
        assert message.report.summary.interrupted_rate == 0.25
        assert message.report.drifts[0].change == 0.5
        assert DashboardSummary(**(SUMMARY.model_dump() | {"calls": 0})).error_rate is None

    @pytest.mark.parametrize(
        ("used", "budget", "expected"),
        [
            (0, 8192, (0.0, False)),
            (900, 1000, (0.9, True)),
            (800, 1000, (0.8, True)),
            (None, 8192, (None, False)),
            (950, None, (None, False)),
        ],
    )
    def test_context_usage_derived_figures(
        self, used: int | None, budget: int | None, expected: tuple[float | None, bool]
    ) -> None:
        usage = ContextUsage(used=used, budget=budget, warning_threshold=0.8)
        question = AskCompression(request_id=uuid4(), used=used, budget=budget, warning_threshold=0.8)

        assert (usage.ratio, usage.near_limit) == expected
        assert (question.ratio, question.near_limit) == expected

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
            '{"type": "request_command", "request_id": "' + CLIENT_ID + '", "command": "rm_rf", "params": {}}',
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
            "unknown-client-command",
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
