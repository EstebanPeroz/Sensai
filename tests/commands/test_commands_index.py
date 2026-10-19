from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from sensai.commands.index import Index
from sensai.rag.indexer import Indexer
from sensai.rag.store import ChunkTable, VectorStore
from tests.commands.fakes import FakeUI
from tests.rag.fakes import FakeIndexation

CATS_THEN_ENGINES = (
    "The cat sleeps all day. It purrs when you pet it. Cats sleep sixteen hours. "
    "The engine burns fuel. The engine efficiency is low. The engine wastes heat."
)


@pytest.fixture
def store() -> VectorStore:
    """A VectorStore backed by an in-memory database."""
    return VectorStore("sqlite:///:memory:")


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
