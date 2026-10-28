from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sensai_protocol import (
    AiChunk,
    AskUser,
    Cancel,
    Chat,
    Command,
    Complete,
    Completions,
    ConfirmAnswer,
    Confirmation,
    ConfirmTool,
    Dashboard,
    DashboardReport,
    DashboardSummary,
    Error,
    History,
    HistoryMessage,
    Input,
    Login,
    Logout,
    Period,
    Role,
    Session,
    System,
    UserAnswer,
)
from sensai_tui.ui import ChatApp
from sensai_tui.ui._dashboard import DashboardScreen
from sensai_tui.ui._dialogs import ConfirmScreen, QuestionScreen
from sensai_tui.ui._textual import ChatItem, CompletionList
from textual.pilot import Pilot
from textual.widgets import Input as InputWidget
from textual.widgets import Label, ListView

from tests.client.tui.fakes import FakeConnection

MODEL = "qwen3:1.7b"


def _session(commands: list[str] | None = None) -> Session:
    return Session(client_id=uuid4(), model=MODEL, commands=commands or ["/help", "/model"], effort="medium")


def _items(app: ChatApp) -> list[ChatItem]:
    return list(app.query_one("#chat", ListView).query(ChatItem))


async def _submit(pilot: Pilot, text: str) -> None:
    field = pilot.app.query_one("#input", InputWidget)
    field.value = text
    await field.action_submit()
    await pilot.pause()


class TestLogin:
    async def test_mount_sends_login(self) -> None:
        conn = FakeConnection()
        async with ChatApp(conn, MODEL).run_test() as pilot:
            await pilot.pause()
            assert conn.sent == [Login(model=MODEL)]

    async def test_session_sets_effort_model_and_commands(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            session = _session()
            conn.push(session)
            await pilot.pause()
            assert app.client_id == session.client_id
            assert app.query_one("#input", InputWidget).border_subtitle == "effort: medium"
            assert str(app.query_one("#header-model", Label).render()) == MODEL

            await pilot.press("/", "m", "o")
            dropdown = app.query_one(CompletionList)
            assert dropdown.display
            assert [str(dropdown.get_option_at_index(i).prompt) for i in range(dropdown.option_count)] == ["/model"]


class TestSubmit:
    async def test_text_sends_chat(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            conn.push(_session())
            await _submit(pilot, "hello")
            assert conn.sent[-1] == Chat(prompt="hello")
            assert _items(app) == []

    async def test_slash_sends_command(self) -> None:
        conn = FakeConnection()
        async with ChatApp(conn, MODEL).run_test() as pilot:
            await _submit(pilot, "/model x")
            assert conn.sent[-1] == Command(command="/model x")

    async def test_empty_sends_nothing(self) -> None:
        conn = FakeConnection()
        async with ChatApp(conn, MODEL).run_test() as pilot:
            await _submit(pilot, "   ")
            assert conn.sent == [Login(model=MODEL)]

    async def test_quit_sends_logout_and_exits(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            await _submit(pilot, "/q")
            await pilot.pause()
            assert conn.sent[-1] == Logout()
            assert conn.closed
        assert app.return_code == 0

    async def test_input_echo_is_rendered(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            conn.push(Input(role=Role.USER, content="hello"), Input(role=Role.USER, content="/help", command=True))
            await pilot.pause()
            assert [(item.role, item.mode, item.text) for item in _items(app)] == [
                ("user", "message", "hello"),
                ("user", "command", "/help"),
            ]


class TestStreaming:
    async def test_chunks_stream_into_one_entry(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            conn.push(AiChunk(content="Hel"), AiChunk(content="lo"), AiChunk(done=True))
            await pilot.pause()
            items = _items(app)
            assert len(items) == 1
            assert (items[0].mode, items[0].text) == ("content", "Hello")

    async def test_thinking_then_content(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            conn.push(AiChunk(thinking="hm"), AiChunk(content="ok"))
            await pilot.pause()
            assert [(item.mode, item.text) for item in _items(app)] == [("thinking", "hm"), ("content", "ok")]

    async def test_input_during_stream_cancels_first(self) -> None:
        conn = FakeConnection()
        async with ChatApp(conn, MODEL).run_test() as pilot:
            conn.push(AiChunk(content="partial"))
            await pilot.pause()
            await _submit(pilot, "next")
            assert conn.sent[-2:] == [Cancel(), Chat(prompt="next")]

    async def test_input_after_stream_does_not_cancel(self) -> None:
        conn = FakeConnection()
        async with ChatApp(conn, MODEL).run_test() as pilot:
            conn.push(AiChunk(content="answer"), AiChunk(done=True))
            await pilot.pause()
            await _submit(pilot, "next")
            assert Cancel() not in conn.sent

    async def test_escape_and_ctrl_c_cancel(self) -> None:
        conn = FakeConnection()
        async with ChatApp(conn, MODEL).run_test() as pilot:
            await pilot.press("escape")
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert conn.sent[-2:] == [Cancel(), Cancel()]

    async def test_chunk_error_is_shown(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            conn.push(AiChunk(error="model crashed", done=True))
            await pilot.pause()
            assert [(item.mode, item.text) for item in _items(app)] == [("error", "model crashed")]


class TestServerMessages:
    async def test_history_replaces_transcript(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            conn.push(Input(role=Role.USER, content="old"))
            conn.push(
                History(
                    messages=[
                        HistoryMessage(uuid=uuid4(), role=Role.USER, content="hi"),
                        HistoryMessage(uuid=None, role=Role.ASSISTANT, content="hello"),
                        HistoryMessage(uuid=None, role=Role.SYSTEM, content="hidden"),
                    ]
                )
            )
            await pilot.pause()
            assert [(item.role, item.text) for item in _items(app)] == [("user", "hi"), ("assistant", "hello")]

    async def test_error_and_system_add_entries(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            conn.push(Error(message="boom"), System(message="done"), System(message="more", append=True))
            await pilot.pause()
            assert [(item.mode, item.text) for item in _items(app)] == [
                ("error", "boom"),
                ("system", "done"),
                ("text", "more"),
            ]

    async def test_completions_fill_the_dropdown(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            conn.push(_session())
            await pilot.pause()
            await pilot.press(*"/model ")
            await pilot.pause()
            assert conn.sent[-1] == Complete(text="/model ")
            conn.push(Completions(text="/model ", values=["llama3:8b", "qwen3:1.7b"]))
            await pilot.pause()
            dropdown = app.query_one(CompletionList)
            assert dropdown.display
            assert dropdown.option_count == 2

    async def test_late_completions_are_dropped(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            conn.push(_session())
            await pilot.pause()
            await pilot.press(*"/model a ")
            conn.push(Completions(text="/model ", values=["late"]))
            await pilot.pause()
            assert not app.query_one(CompletionList).display

    async def test_disconnect_stops_input(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            conn.disconnect()
            await pilot.pause()
            assert _items(app)[-1].text == "disconnected from server"
            assert app.query_one("#input", InputWidget).disabled
            await _submit(pilot, "hello")
            assert conn.sent == [Login(model=MODEL)]


def _report() -> DashboardReport:
    summary = DashboardSummary(
        calls=0,
        errors=0,
        interrupted=0,
        latency_p50_ms=None,
        latency_p95_ms=None,
        first_token_p50_ms=None,
        tokens_per_s=None,
        prompt_tokens=0,
        completion_tokens=0,
    )
    return DashboardReport(
        period=Period.DAY,
        start=datetime(2026, 10, 13, 12, 0, tzinfo=UTC),
        end=datetime(2026, 10, 14, 12, 0, tzinfo=UTC),
        summary=summary,
        calls_series=[0] * 24,
        latency_series=[None] * 24,
        groups=[],
        drifts=[],
        recent_errors=[],
        baseline_periods=7,
    )


class TestConfirmTool:
    @pytest.mark.parametrize(
        ("key", "answer"),
        [
            ("y", Confirmation.ALLOW),
            ("a", Confirmation.ALWAYS),
            ("n", Confirmation.DENY),
            ("escape", Confirmation.DENY),
        ],
    )
    async def test_key_sends_its_answer(self, key: str, answer: Confirmation) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            request = ConfirmTool(request_id=uuid4(), tool_name="web_search", arguments={"query": "paris"})
            conn.push(_session(), request)
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)

            await pilot.press(key)
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmScreen)
            assert conn.sent[-1] == ConfirmAnswer(request_id=request.request_id, answer=answer)
            assert Cancel() not in conn.sent

    async def test_hides_the_completions(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            conn.push(_session())
            await pilot.press("/", "m")
            assert app.query_one(CompletionList).display
            conn.push(ConfirmTool(request_id=uuid4(), tool_name="now", arguments={}))
            await pilot.pause()
            assert not app.query_one(CompletionList).display

    async def test_disconnect_closes_the_dialog(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            conn.push(ConfirmTool(request_id=uuid4(), tool_name="now", arguments={}))
            await pilot.pause()
            conn.disconnect()
            await pilot.pause()
            assert len(app.screen_stack) == 1
            assert conn.sent == [Login(model=MODEL)]


class TestAskUser:
    async def test_answer_is_sent_back_not_chatted(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            request = AskUser(request_id=uuid4(), question="Which city?")
            conn.push(_session(), request)
            await pilot.pause()
            assert isinstance(app.screen, QuestionScreen)

            await pilot.press(*"/Paris", "enter")
            await pilot.pause()
            assert not isinstance(app.screen, QuestionScreen)
            assert conn.sent[-1] == UserAnswer(request_id=request.request_id, answer="/Paris")
            assert not any(isinstance(message, (Chat, Command, Complete)) for message in conn.sent)
            assert app.query_one("#input", InputWidget).value == ""

    async def test_escape_skips_the_question(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            request = AskUser(request_id=uuid4(), question="Which city?")
            conn.push(request)
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert conn.sent[-1] == UserAnswer(request_id=request.request_id, answer=None)
            assert Cancel() not in conn.sent


class TestDashboard:
    @pytest.mark.parametrize("key", ["q", "escape"])
    async def test_opens_over_the_chat_and_closes_back_to_it(self, key: str) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test(size=(140, 60)) as pilot:
            conn.push(Dashboard(report=_report()))
            await pilot.pause()
            assert isinstance(app.screen, DashboardScreen)

            await pilot.press(key)
            await pilot.pause()
            assert not isinstance(app.screen, DashboardScreen)
            assert isinstance(app.focused, InputWidget)
            assert Cancel() not in conn.sent
