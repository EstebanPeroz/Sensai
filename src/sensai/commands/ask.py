import threading
from typing import TYPE_CHECKING, override

from sensai.commands.base import Command
from sensai.error import SensaiError

if TYPE_CHECKING:
    from sensai.rag.retrievial_rag import Retrieval

# How often the dispatch thread looks for a UI event while an answer is being built.
_POLL_SECONDS = 0.05


class Ask(Command):
    """Ask command used to answer a question from the indexed documents."""

    name = "ask"
    help = "Answer a question from the documents indexed with /index"
    usage = "/ask <question>"

    _retrieval: Retrieval

    def __init__(self, retrieval: Retrieval) -> None:
        """Init the ask Command with the retrieval answering the questions."""
        super().__init__()
        self._retrieval = retrieval
        self._running = threading.Event()

    @override
    def execute(self, *args: str) -> None:
        """Answer the question, off the thread that dispatches the UI events.

        Embedding the question and waiting for the model takes seconds. Done on
        the dispatch thread it would hold back every event meanwhile, the one
        asking to quit included.
        """
        question = " ".join(args).strip()
        if not question:
            self._ui.send_system_message(f"Usage: {self.usage}")
            return
        if self._running.is_set():
            self._ui.send_system_message("A question is already being answered.")
            return
        self._running.set()
        threading.Thread(target=self._answer, args=(question,), daemon=True).start()
        while self._running.is_set():
            if self._ui.wait_event(timeout=_POLL_SECONDS):
                return

    def _answer(self, question: str) -> None:
        """Answer `question` and report it, sources included. Runs on a worker thread."""
        try:
            answer, sources = self._retrieval.answer(question)
            self._ui.send_system_message(answer)
            if sources:
                self._ui.send_system_message(f"Sources: {', '.join(sources)}")
        except SensaiError as error:
            self._ui.send_system_message(error.message)
        except Exception as error:  # noqa: BLE001 - nothing above would report a worker dying
            self._ui.send_system_message(f"The question could not be answered: {error!r}")
        finally:
            self._running.clear()
