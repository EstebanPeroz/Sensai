import time
from pathlib import Path
from typing import override

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from sensai.commands.base import Command
from sensai.commands.index import Index
from sensai.rag.indexation_rag import ChunkStrategy, Indexation
from sensai.rag.indexer import Indexer
from sensai.rag.store import ChunkTable, VectorStore
from tests.commands.fakes import FakeUI
from tests.rag.fakes import BlockingEmbedder, FakeIndexation, FlakyEmbedder

CATS_THEN_ENGINES = (
    "The cat sleeps all day. It purrs when you pet it. Cats sleep sixteen hours. "
    "The engine burns fuel. The engine efficiency is low. The engine wastes heat."
)


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    """A VectorStore on a fresh sqlite file: the indexation writes from a worker thread."""
    return VectorStore(f"sqlite:///{tmp_path / 'rag.db'}")


@pytest.fixture
def command(store: VectorStore, fake_ui: FakeUI) -> Index:
    """An Index command writing to `store`, with its UI already bound."""
    del fake_ui  # the fixture binds the UI every command shares
    return Index(Indexer(FakeIndexation(), store))


def write(directory: Path, name: str, content: str = CATS_THEN_ENGINES) -> str:
    """Write `content` to `name` under `directory` and return its path."""
    source = directory / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(content, encoding="utf-8")
    return str(source)


def sources(store: VectorStore) -> list[str]:
    """The source of every stored chunk."""
    with Session(store._engine) as session:  # noqa: SLF001
        return list(session.scalars(select(ChunkTable.source)))


class TestExecute:
    def test_without_an_argument_it_shows_the_usage(self, command: Index, fake_ui: FakeUI) -> None:
        command.execute()

        assert fake_ui.messages == [f"Usage: {command.usage}"]

    def test_one_source_is_reported_without_a_summary(self, command: Index, fake_ui: FakeUI, tmp_path: Path) -> None:
        source = write(tmp_path, "a.txt")

        command.execute(source)

        assert fake_ui.messages == [f"{source}: 2 chunks"]

    def test_several_sources_are_each_reported_then_summarised(
        self, command: Index, fake_ui: FakeUI, tmp_path: Path
    ) -> None:
        first = write(tmp_path, "a.txt")
        second = write(tmp_path, "b.txt", "The cat sleeps.")

        command.execute(first, second)

        assert fake_ui.messages == [
            f"{first}: 2 chunks",
            f"{second}: 1 chunks",
            "Indexed 2/2 sources into 3 chunks",
        ]

    def test_every_source_reaches_the_store(self, command: Index, store: VectorStore, tmp_path: Path) -> None:
        first = write(tmp_path, "a.txt")
        second = write(tmp_path, "b.txt", "The cat sleeps.")

        command.execute(first, second)

        assert sorted(set(sources(store))) == sorted([first, second])


class TestExecuteWithFailures:
    def test_a_bad_source_does_not_stop_the_others(self, command: Index, fake_ui: FakeUI, tmp_path: Path) -> None:
        good = write(tmp_path, "a.txt", "The cat sleeps.")
        missing = str(tmp_path / "absent.md")
        other = write(tmp_path, "b.txt", "The engine burns fuel.")

        command.execute(good, missing, other)

        assert f"{good}: 1 chunks" in fake_ui.messages
        assert f"{other}: 1 chunks" in fake_ui.messages

    def test_the_failure_is_reported_with_its_reason(self, command: Index, fake_ui: FakeUI, tmp_path: Path) -> None:
        missing = str(tmp_path / "absent.md")

        command.execute(missing)

        assert any("SourceNotFoundError" in message for message in fake_ui.messages)

    def test_the_summary_counts_what_was_skipped(self, command: Index, fake_ui: FakeUI, tmp_path: Path) -> None:
        good = write(tmp_path, "a.txt", "The cat sleeps.")

        command.execute(good, str(tmp_path / "absent.md"), write(tmp_path, "b.docx", "data"))

        assert fake_ui.messages[-1] == "Indexed 1/3 sources into 1 chunks (2 skipped)"

    def test_a_single_failing_source_still_gets_a_summary(
        self, command: Index, fake_ui: FakeUI, tmp_path: Path
    ) -> None:
        command.execute(str(tmp_path / "absent.md"))

        assert fake_ui.messages[-1] == "Indexed 0/1 sources into 0 chunks (1 skipped)"

    def test_a_failing_source_stores_nothing(self, command: Index, store: VectorStore, tmp_path: Path) -> None:
        command.execute(str(tmp_path / "absent.md"))

        assert sources(store) == []


class TestCompletions:
    def test_offers_the_indexable_files(self, command: Index, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        write(tmp_path, "notes.md", "text")
        write(tmp_path, "page.html", "<p>text</p>")
        monkeypatch.chdir(tmp_path)

        assert sorted(command.completions([])) == ["notes.md", "page.html"]

    def test_ignores_the_formats_it_cannot_read(
        self, command: Index, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write(tmp_path, "notes.md", "text")
        write(tmp_path, "archive.zip", "data")
        monkeypatch.chdir(tmp_path)

        assert command.completions([]) == ["notes.md"]

    def test_walks_the_subdirectories(self, command: Index, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        write(tmp_path, "docs/deep/notes.md", "text")
        monkeypatch.chdir(tmp_path)

        assert command.completions([]) == ["docs/deep/notes.md"]

    def test_prunes_hidden_directories(self, command: Index, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        write(tmp_path, ".venv/noise.md", "text")
        monkeypatch.chdir(tmp_path)

        assert command.completions([]) == []

    def test_prunes_the_caches(self, command: Index, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        write(tmp_path, "__pycache__/noise.md", "text")
        monkeypatch.chdir(tmp_path)

        assert command.completions([]) == []

    def test_offers_the_same_files_at_every_position(
        self, command: Index, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write(tmp_path, "notes.md", "text")
        monkeypatch.chdir(tmp_path)

        assert command.completions([]) == command.completions(["notes.md"]) == command.completions(["a", "b"])

    def test_the_list_is_capped(self, command: Index, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for i in range(60):
            write(tmp_path, f"notes{i:02d}.md", "text")
        monkeypatch.chdir(tmp_path)

        assert len(command.completions([])) == 50


class TestExecuteWithProviderFailures:
    @pytest.fixture
    def flaky(self, store: VectorStore, fake_ui: FakeUI) -> Index:
        """An Index command whose embedder fails on any text saying "boom"."""
        del fake_ui
        return Index(Indexer(Indexation(FlakyEmbedder()), store))

    def test_a_transport_failure_does_not_abort_the_later_sources(
        self, flaky: Index, fake_ui: FakeUI, tmp_path: Path
    ) -> None:
        first = write(tmp_path, "a.txt", "The cat sleeps.")
        boom = write(tmp_path, "b.txt", "This one says boom.")
        last = write(tmp_path, "c.txt", "The engine burns fuel.")

        flaky.execute(first, boom, last)

        assert f"{first}: 1 chunks" in fake_ui.messages
        assert f"{last}: 1 chunks" in fake_ui.messages

    def test_the_failing_source_is_reported_as_an_embedding_error(
        self, flaky: Index, fake_ui: FakeUI, tmp_path: Path
    ) -> None:
        boom = write(tmp_path, "b.txt", "This one says boom.")

        flaky.execute(boom)

        assert any("EmbeddingError" in message for message in fake_ui.messages)

    def test_the_summary_counts_it_as_skipped(self, flaky: Index, fake_ui: FakeUI, tmp_path: Path) -> None:
        good = write(tmp_path, "a.txt", "The cat sleeps.")
        boom = write(tmp_path, "b.txt", "This one says boom.")

        flaky.execute(good, boom)

        assert fake_ui.messages[-1] == "Indexed 1/2 sources into 1 chunks (1 skipped)"

    def test_the_sources_that_worked_are_still_stored(self, flaky: Index, store: VectorStore, tmp_path: Path) -> None:
        good = write(tmp_path, "a.txt", "The cat sleeps.")
        boom = write(tmp_path, "b.txt", "This one says boom.")

        flaky.execute(boom, good)

        assert sources(store) == [good]


class BrokenIndexer(Indexer):
    """An Indexer that fails the way a bug does, not the way a bad source does."""

    @override
    def index(self, source: str, *, strategy: ChunkStrategy = ChunkStrategy.AUTO) -> int:
        """Raise an error no caller expects."""
        del source, strategy
        raise ZeroDivisionError


class EventPendingUI(FakeUI):
    """A FakeUI that always reports a queued event, as one does when the user acts."""

    @override
    def wait_event(self, *, timeout: float | None = None) -> bool:
        del timeout
        return True


class TestExecuteOffTheDispatchThread:
    def test_it_returns_as_soon_as_an_event_is_queued(self, store: VectorStore, tmp_path: Path) -> None:
        ui = EventPendingUI()
        Command.bind_ui(ui)
        command = Index(Indexer(FakeIndexation(), store))

        command.execute(write(tmp_path, "a.txt"))

        assert ui.messages == []  # the worker reports later, the dispatch loop gets its event now

    def test_the_work_still_finishes_after_it_returned(self, store: VectorStore, tmp_path: Path) -> None:
        ui = EventPendingUI()
        Command.bind_ui(ui)
        command = Index(Indexer(FakeIndexation(), store))
        source = write(tmp_path, "a.txt")

        command.execute(source)

        deadline = time.monotonic() + 5
        while not sources(store) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sources(store) == [source, source]

    def test_a_second_indexation_is_refused_while_one_runs(
        self, store: VectorStore, tmp_path: Path, fake_ui: FakeUI
    ) -> None:
        del fake_ui
        ui = EventPendingUI()
        Command.bind_ui(ui)
        command = Index(Indexer(Indexation(BlockingEmbedder()), store))
        command.execute(write(tmp_path, "a.txt"))

        command.execute(write(tmp_path, "b.txt"))

        assert "An indexation is already running." in ui.messages

    def test_the_guard_is_released_once_done(self, command: Index, fake_ui: FakeUI, tmp_path: Path) -> None:
        source = write(tmp_path, "a.txt")

        command.execute(source)
        fake_ui.messages.clear()
        command.execute(source)

        assert "An indexation is already running." not in fake_ui.messages

    def test_an_unexpected_error_is_reported_instead_of_vanishing(
        self, store: VectorStore, fake_ui: FakeUI, tmp_path: Path
    ) -> None:
        command = Index(BrokenIndexer(FakeIndexation(), store))

        command.execute(write(tmp_path, "a.txt"))

        assert any("unexpected error" in message for message in fake_ui.messages)

    def test_the_guard_is_released_even_when_the_indexer_breaks(
        self, store: VectorStore, fake_ui: FakeUI, tmp_path: Path
    ) -> None:
        command = Index(BrokenIndexer(FakeIndexation(), store))
        command.execute(write(tmp_path, "a.txt"))

        command.execute(write(tmp_path, "b.txt"))

        assert "An indexation is already running." not in fake_ui.messages
