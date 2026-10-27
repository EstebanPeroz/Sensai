from uuid import uuid4

from sensai_protocol import (
    AiChunk,
    Cancel,
    Chat,
    Command,
    Complete,
    Completions,
    Error,
    History,
    HistoryMessage,
    Input,
    Login,
    Logout,
    Role,
    Session,
    System,
)
from sensai_tui.ui import ChatApp
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
