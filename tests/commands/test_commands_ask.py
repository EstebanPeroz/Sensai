import time
from collections.abc import Iterator
from pathlib import Path
from typing import override

import pytest

from sensai.commands.ask import Ask
from sensai.commands.base import Command
from sensai.rag.collection import Collection
from sensai.rag.indexation_rag import Indexation
from sensai.rag.retrievial_rag import Retrieval
from tests.commands.fakes import FakeUI
from tests.rag.fakes import BlockingEmbedder, FailingEmbedder, FakeAnswerer, FakeIndexation
from tests.rag.seeding import seed

ABOUT_ENGINES = "The engine burns fuel."
ABOUT_CATS = "The cat sleeps."


@pytest.fixture
def collection(tmp_path: Path) -> Iterator[Collection]:
    """A Collection on a fresh embedded database, closed once the test is done."""
    opened = Collection.open(str(tmp_path / "qdrant"))
    yield opened
    opened.close()


@pytest.fixture
def answerer() -> FakeAnswerer:
    """A provider answering a canned sentence, with no server behind it."""
    return FakeAnswerer()


@pytest.fixture
def command(collection: Collection, answerer: FakeAnswerer, fake_ui: FakeUI) -> Ask:
    """An Ask command over a collection holding one chunk on engines and one on cats."""
    del fake_ui  # the fixture binds the UI every command shares
    indexation = FakeIndexation()
    for source, text in (("engines.md", ABOUT_ENGINES), ("cats.md", ABOUT_CATS)):
        seed(collection, indexation, source, text)
    return Ask(Retrieval(indexation, collection, answerer))


class EventPendingUI(FakeUI):
    """A FakeUI that always reports a queued event, as one does when the user acts."""

    @override
    def wait_event(self, *, timeout: float | None = None) -> bool:
        del timeout
        return True


class TestExecute:
    def test_without_a_question_it_shows_the_usage(self, command: Ask, fake_ui: FakeUI) -> None:
        command.execute()

        assert fake_ui.messages == [f"Usage: {command.usage}"]

    def test_a_blank_question_shows_the_usage(self, command: Ask, fake_ui: FakeUI) -> None:
        command.execute("   ")

        assert fake_ui.messages == [f"Usage: {command.usage}"]

    def test_it_answers_and_names_the_sources(self, command: Ask, fake_ui: FakeUI) -> None:
        command.execute("What", "about", "the", "engine", "?")

        assert fake_ui.messages == ["A grounded answer.", "Sources: engines.md"]

    def test_the_words_are_joined_back_into_one_question(self, command: Ask, answerer: FakeAnswerer) -> None:
        command.execute("What", "about", "the", "engine", "?")

        assert answerer.payloads[0]["messages"][-1]["content"] == "What about the engine ?"

    def test_the_retrieved_chunk_reaches_the_model(self, command: Ask, answerer: FakeAnswerer) -> None:
        command.execute("What", "about", "the", "engine", "?")

        assert ABOUT_ENGINES in answerer.payloads[0]["messages"][0]["content"]

    def test_an_empty_store_is_said_plainly_without_sources(
        self, collection: Collection, answerer: FakeAnswerer, fake_ui: FakeUI
    ) -> None:
        command = Ask(Retrieval(FakeIndexation(), collection, answerer))

        command.execute("anything")

        assert len(fake_ui.messages) == 1
        assert "nothing close enough" in fake_ui.messages[0]

    def test_an_empty_store_asks_the_model_nothing(
        self, collection: Collection, answerer: FakeAnswerer, fake_ui: FakeUI
    ) -> None:
        del fake_ui
        Ask(Retrieval(FakeIndexation(), collection, answerer)).execute("anything")

        assert answerer.payloads == []


class TestExecuteFailures:
    def test_an_embedding_failure_is_reported(
        self, collection: Collection, answerer: FakeAnswerer, fake_ui: FakeUI
    ) -> None:
        command = Ask(Retrieval(Indexation(FailingEmbedder()), collection, answerer))

        command.execute("anything")

        assert any("EmbeddingError" in message for message in fake_ui.messages)

    def test_a_failure_does_not_reach_the_model(
        self, collection: Collection, answerer: FakeAnswerer, fake_ui: FakeUI
    ) -> None:
        del fake_ui
        command = Ask(Retrieval(Indexation(FailingEmbedder()), collection, answerer))

        command.execute("anything")

        assert answerer.payloads == []


class TestExecuteOffTheDispatchThread:
    def test_it_returns_while_the_work_is_still_running(self, collection: Collection) -> None:
        ui = EventPendingUI()
        Command.bind_ui(ui)
        command = Ask(Retrieval(Indexation(BlockingEmbedder()), collection, FakeAnswerer()))

        command.execute("anything")

        assert ui.messages == []  # the dispatch loop gets its event back before the answer lands

    def test_the_work_still_finishes_after_it_returned(self, collection: Collection) -> None:
        ui = EventPendingUI()
        Command.bind_ui(ui)
        command = Ask(Retrieval(FakeIndexation(), collection, FakeAnswerer()))

        command.execute("anything")

        deadline = time.monotonic() + 5
        while not ui.messages and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ui.messages

    def test_a_second_question_is_refused_while_one_runs(self, collection: Collection, fake_ui: FakeUI) -> None:
        del fake_ui
        ui = EventPendingUI()
        Command.bind_ui(ui)
        command = Ask(Retrieval(Indexation(BlockingEmbedder()), collection, FakeAnswerer()))
        command.execute("first")

        command.execute("second")

        assert ui.messages == ["A question is already being answered."]

    def test_the_guard_is_released_once_done(self, command: Ask, fake_ui: FakeUI) -> None:
        command.execute("engine")
        fake_ui.messages.clear()

        command.execute("engine")

        assert "A question is already being answered." not in fake_ui.messages
