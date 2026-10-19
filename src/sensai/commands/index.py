import threading
from pathlib import Path
from typing import TYPE_CHECKING, override

from sensai.commands.base import Command
from sensai.rag.error import RagError

if TYPE_CHECKING:
    from sensai.rag.indexer import Indexer

_INDEXABLE_SUFFIXES = (".txt", ".md", ".pdf", ".html", ".htm")
_IGNORED_DIRECTORIES = {"__pycache__", "node_modules"}
_MAX_COMPLETIONS = 50
# How often the dispatch thread looks for a UI event while indexing runs.
_POLL_SECONDS = 0.05


class Index(Command):
    """Index command used to add documents to the RAG."""

    name = "index"
    help = "Index documents into the RAG, from paths or URLs"
    usage = "/index <path|url> [path|url ...]"

    _indexer: Indexer

    def __init__(self, indexer: Indexer) -> None:
        """Init the index Command with the indexer writing to the store."""
        super().__init__()
        self._indexer = indexer
        self._running = threading.Event()

    @override
    def execute(self, *args: str) -> None:
        """Index every given source, off the thread that dispatches the UI events.

        Embedding a document takes seconds, and minutes for a large one. Done on
        the dispatch thread it would hold back every event meanwhile, the one
        asking to quit included. So the work goes to a worker thread, and this
        waits on it while still watching for an event: when one shows up, it
        returns and leaves the event to the dispatch loop, the indexation
        carrying on and reporting on its own.
        """
        if not args:
            self._ui.send_system_message(f"Usage: {self.usage}")
            return
        if self._running.is_set():
            self._ui.send_system_message("An indexation is already running.")
            return
        self._running.set()
        threading.Thread(target=self._index_all, args=args, daemon=True).start()
        while self._running.is_set():
            if self._ui.wait_event(timeout=_POLL_SECONDS):
                return

    def _index_all(self, *sources: str) -> None:
        """Index every source, reporting each one then a summary. Runs on a worker thread.

        A source that cannot be indexed is reported and skipped: the others are
        still indexed, so one bad path does not waste the whole batch.
        """
        try:
            chunks, failures = 0, 0
            for source in sources:
                try:
                    indexed = self._indexer.index(source)
                except RagError as error:
                    failures += 1
                    self._ui.send_system_message(f"{source}: {error.message}")
                    continue
                chunks += indexed
                self._ui.send_system_message(f"{source}: {indexed} chunks")
            if len(sources) > 1 or failures:
                self._ui.send_system_message(self._summary(len(sources), chunks, failures))
        except Exception as error:  # noqa: BLE001 - nothing above would report a worker dying
            self._ui.send_system_message(f"Indexation stopped on an unexpected error: {error!r}")
        finally:
            self._running.clear()

    @staticmethod
    def _summary(sources: int, chunks: int, failures: int) -> str:
        """One line telling how many sources were indexed, and how many were skipped."""
        indexed = sources - failures
        line = f"Indexed {indexed}/{sources} sources into {chunks} chunks"
        return line if not failures else f"{line} ({failures} skipped)"

    @override
    def completions(self, args: list[str]) -> list[str]:
        """Offer the indexable files under the working directory, at every argument position."""
        del args
        return self._indexable_files()

    @staticmethod
    def _indexable_files() -> list[str]:
        """Relative paths of the indexable files under the working directory.

        Hidden directories and caches are pruned before being walked, so a
        virtualenv does not get crawled, and the list is capped to stay usable.
        """
        found: list[str] = []
        for directory, subdirectories, files in Path().walk():
            subdirectories[:] = [
                name for name in subdirectories if not name.startswith(".") and name not in _IGNORED_DIRECTORIES
            ]
            for name in sorted(files):
                if Path(name).suffix.lower() in _INDEXABLE_SUFFIXES:
                    found.append(str(directory / name))
                    if len(found) >= _MAX_COMPLETIONS:
                        return found
        return found
