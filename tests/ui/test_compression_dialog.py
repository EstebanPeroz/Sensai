import asyncio

import pytest
from textual.widgets import Button

from sensai.monitoring.budget import ContextUsage
from sensai.ui.adapter import CompressionChoice
from sensai.ui.event_queue import EventQueue
from sensai.ui.tui._dialogs import CompressionScreen
from sensai.ui.tui._textual import ChatApp

USAGE = ContextUsage(900, 1000, 0.8)


def answer_with(key: str | None = None, button: str | None = None) -> list[CompressionChoice | None]:
    """Open the dialog over the chat, answer it with `key` or a click on `button`, and return what it gave."""
    app = ChatApp(EventQueue())
    answers: list[CompressionChoice | None] = []

    async def scenario() -> None:
        async with app.run_test() as pilot:
            app.open_dialog(CompressionScreen(USAGE), answers.append)
            await pilot.pause()
            assert isinstance(app.screen, CompressionScreen)
            if key is not None:
                await pilot.press(key)
            if button is not None:
                await pilot.click(f"#{button}")
            await pilot.pause()
            assert not isinstance(app.screen, CompressionScreen)

    asyncio.run(scenario())
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
    def test_key_answers(self, key: str, expected: CompressionChoice) -> None:
        assert answer_with(key=key) == [expected]

    @pytest.mark.parametrize(
        ("button", "expected"),
        [("compress", CompressionChoice.COMPRESS), ("new", CompressionChoice.NEW_CONVERSATION)],
    )
    def test_button_answers(self, button: str, expected: CompressionChoice) -> None:
        assert answer_with(button=button) == [expected]

    def test_offers_the_two_choices(self) -> None:
        app = ChatApp(EventQueue())

        async def scenario() -> None:
            async with app.run_test():
                app.open_dialog(CompressionScreen(USAGE), lambda _answer: None)
                await asyncio.sleep(0.1)
                labels = [str(button.label) for button in app.screen.query(Button)]
                assert labels == ["Compress and continue (c)", "New conversation (n)"]

        asyncio.run(scenario())
