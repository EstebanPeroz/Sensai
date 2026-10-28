import asyncio

import pytest
from textual.widgets import Input

from sensai.monitoring.budget import ContextUsage
from sensai.ui.event_queue import EventQueue
from sensai.ui.tui._textual import ChatApp, _format_usage


class TestFormatUsage:
    @pytest.mark.parametrize(
        ("usage", "expected"),
        [
            (ContextUsage(0, 8192, 0.8), "context: 0 / 8.2k (0%)"),
            (ContextUsage(2048, 8192, 0.8), "context: 2.0k / 8.2k (25%)"),
            (ContextUsage(950, None, 0.8), "context: 950"),
        ],
    )
    def test_shows_the_usage_against_the_budget(self, usage: ContextUsage, expected: str) -> None:
        assert _format_usage(usage) == expected

    def test_turns_red_near_the_limit(self) -> None:
        assert _format_usage(ContextUsage(7000, 8192, 0.8)) == "[red]context: 7.0k / 8.2k (85%)[/red]"


class TestChatAppContextUsage:
    def test_is_shown_on_the_input_box(self) -> None:
        app = ChatApp(EventQueue())

        async def scenario() -> None:
            async with app.run_test() as pilot:
                app.update_context_usage(ContextUsage(2048, 8192, 0.8))
                await pilot.pause()
                assert app.query_one("#input", Input).border_title == "context: 2.0k / 8.2k (25%)"

        asyncio.run(scenario())
