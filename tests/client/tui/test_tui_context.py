from uuid import uuid4

import pytest
from sensai_protocol import AskCompression, Cancel, CompressionAnswer, CompressionChoice, ContextUsage, Login
from sensai_tui.ui import ChatApp
from sensai_tui.ui._dialogs import CompressionScreen
from sensai_tui.ui._textual import CompletionList, _format_usage
from textual.widgets import Button
from textual.widgets import Input as InputWidget

from tests.client.tui.fakes import FakeConnection

MODEL = "qwen3:1.7b"
QUESTION = AskCompression(request_id=uuid4(), used=900, budget=1000, warning_threshold=0.8)


def _usage(used: int | None, budget: int | None) -> ContextUsage:
    return ContextUsage(used=used, budget=budget, warning_threshold=0.8)


class TestFormatUsage:
    @pytest.mark.parametrize(
        ("usage", "expected"),
        [
            (_usage(0, 8192), "context: 0 / 8.2k (0%)"),
            (_usage(2048, 8192), "context: 2.0k / 8.2k (25%)"),
            (_usage(950, None), "context: 950"),
            (_usage(None, 8192), "context: ? / 8.2k"),
            (_usage(None, None), "context: ?"),
        ],
    )
    def test_shows_the_usage_against_the_budget(self, usage: ContextUsage, expected: str) -> None:
        assert _format_usage(usage) == expected

    def test_turns_red_near_the_limit(self) -> None:
        assert _format_usage(_usage(7000, 8192)) == "[red]context: 7.0k / 8.2k (85%)[/red]"


class TestContextUsage:
    async def test_is_shown_on_the_input_box(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            conn.push(_usage(2048, 8192))
            await pilot.pause()
            assert app.query_one("#input", InputWidget).border_title == "context: 2.0k / 8.2k (25%)"


async def _answer_with(key: str | None = None, button: str | None = None) -> list[CompressionChoice | None]:
    """Open the dialog over the chat, answer it with `key` or a click on `button`, and return what it gave."""
    app = ChatApp(FakeConnection(), MODEL)
    answers: list[CompressionChoice | None] = []
    async with app.run_test() as pilot:
        app.open_dialog(CompressionScreen(QUESTION), answers.append)
        await pilot.pause()
        assert isinstance(app.screen, CompressionScreen)
        if key is not None:
            await pilot.press(key)
        if button is not None:
            await pilot.click(f"#{button}")
        await pilot.pause()
        assert not isinstance(app.screen, CompressionScreen)
    return answers


class TestCompressionScreen:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("c", CompressionChoice.COMPRESS),
            ("n", CompressionChoice.NEW_CONVERSATION),
            ("escape", CompressionChoice.COMPRESS),
        ],
    )
    async def test_key_answers(self, key: str, expected: CompressionChoice) -> None:
        assert await _answer_with(key=key) == [expected]

    @pytest.mark.parametrize(
        ("button", "expected"),
        [("compress", CompressionChoice.COMPRESS), ("new", CompressionChoice.NEW_CONVERSATION)],
    )
    async def test_button_answers(self, button: str, expected: CompressionChoice) -> None:
        assert await _answer_with(button=button) == [expected]

    async def test_offers_the_two_choices(self) -> None:
        app = ChatApp(FakeConnection(), MODEL)
        async with app.run_test() as pilot:
            app.open_dialog(CompressionScreen(QUESTION), lambda _answer: None)
            await pilot.pause()
            labels = [str(button.label) for button in app.screen.query(Button)]
            assert labels == ["Compress and continue (c)", "New conversation (n)"]


class TestAskCompression:
    @pytest.mark.parametrize(
        ("key", "answer"),
        [
            ("c", CompressionChoice.COMPRESS),
            ("n", CompressionChoice.NEW_CONVERSATION),
            ("escape", CompressionChoice.COMPRESS),
        ],
    )
    async def test_key_sends_its_answer(self, key: str, answer: CompressionChoice) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            conn.push(QUESTION)
            await pilot.pause()
            assert isinstance(app.screen, CompressionScreen)

            await pilot.press(key)
            await pilot.pause()
            assert not isinstance(app.screen, CompressionScreen)
            assert conn.sent[-1] == CompressionAnswer(request_id=QUESTION.request_id, answer=answer)
            assert Cancel() not in conn.sent

    async def test_hides_the_completions(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            await pilot.press("/", "m")
            conn.push(QUESTION)
            await pilot.pause()
            assert not app.query_one(CompletionList).display

    async def test_disconnect_closes_the_dialog(self) -> None:
        conn = FakeConnection()
        app = ChatApp(conn, MODEL)
        async with app.run_test() as pilot:
            conn.push(QUESTION)
            await pilot.pause()
            conn.disconnect()
            await pilot.pause()
            assert len(app.screen_stack) == 1
            assert conn.sent == [Login(model=MODEL)]
