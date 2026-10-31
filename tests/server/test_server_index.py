import asyncio
from collections.abc import Iterator
from email.message import Message
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from qdrant_client import QdrantClient
from sensai_protocol import CommandContext, RequestCommand, ServerMessage, System
from sensai_server.commands import CommandRegistry, build_commands
from sensai_server.commands._index import Index
from sensai_server.llm.persona import PersonaManager
from sensai_server.llm.provider import ProviderManager
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai_server.rag import Rag, sources
from sensai_server.rag.collection import Collection
from sensai_server.session import Session, SessionRegistry
from sensai_server.tools.tool_manager import ToolManager

from tests.server.conftest import MODEL, SETTINGS, db_url
from tests.server.fakes import FakeEmbedder, RecordingOutbound

ENGINES = "# Engines\n\nThe engine burns fuel."
CATS = "# Cats\n\nCats sleep all day."
BLOCKED = "blocked.md"


class FilesOutbound(RecordingOutbound):
    """A client holding `files`, by path: a `str` is the file's text, an `Exception` the reason it cannot be read.

    Reading `blocked.md` never gets an answer: `reading_blocked` is set, and the request waits until it is cancelled.
    """

    def __init__(self, files: dict[str, str | Exception]) -> None:
        super().__init__()
        self.files = files
        self.reading_blocked = asyncio.Event()

    async def send(self, message: ServerMessage) -> bool:
        await super().send(message)
        if isinstance(message, RequestCommand):
            path = message.params["path"]
            if path == BLOCKED:
                self.reading_blocked.set()
                return True
            file = self.files[path]
            ctx, error = (None, str(file)) if isinstance(file, Exception) else ({"content": file}, None)
            self.resolve(
                message.request_id,
                CommandContext(request_id=message.request_id, command=message.command, ctx=ctx, error=error),
            )
        return True

    def system_messages(self) -> list[str]:
        return [message.message for message in self.sent if isinstance(message, System)]

    def requests(self) -> list[RequestCommand]:
        return [message for message in self.sent if isinstance(message, RequestCommand)]

    def paths_read(self) -> list[str]:
        return [request.params["path"] for request in self.requests()]


@pytest.fixture
def db(tmp_path: Path) -> SqlAlchemy:
    return SqlAlchemy(db_url(tmp_path))


@pytest.fixture
def rag(db: SqlAlchemy) -> Iterator[Rag]:
    built = Rag(":memory:", FakeEmbedder(), db)
    try:
        yield built
    finally:
        client_of(built).close()


@pytest.fixture
def sessions(db: SqlAlchemy) -> SessionRegistry:
    manager = PersonaManager(db)
    manager.create_default_persona()
    return SessionRegistry(ProviderManager(SETTINGS.llm), manager, ToolManager())


def client_of(rag: Rag) -> QdrantClient:
    return rag._client  # noqa: SLF001


def texts(rag: Rag, rag_id: UUID | None) -> list[str]:
    """The texts stored in the collection of `rag_id`."""
    assert rag_id is not None
    return sorted(payload["text"] for payload in Collection(client_of(rag), rag_id).payloads())


async def open_session(sessions: SessionRegistry, files: dict[str, str | Exception]) -> tuple[Session, FilesOutbound]:
    out = FilesOutbound(files)
    session = await sessions.open(out, MODEL)
    assert session is not None
    out.sent.clear()
    return session, out


class TestIndex:
    async def test_without_argument_it_shows_the_usage(self, rag: Rag, sessions: SessionRegistry) -> None:
        session, out = await open_session(sessions, {})

        await Index(rag).execute(session)

        assert out.system_messages() == [f"Usage: {Index.usage}"]

    async def test_a_client_file_goes_to_the_session_collection_only(self, rag: Rag, sessions: SessionRegistry) -> None:
        session, out = await open_session(sessions, {"engines.md": ENGINES})
        other, _ = await open_session(sessions, {})

        await Index(rag).execute(session, "engines.md")

        assert out.system_messages() == ["engines.md: 1 chunks"]
        assert texts(rag, session.conv.rag_id) == ["Engines\n\nThe engine burns fuel."]
        assert not client_of(rag).collection_exists(str(other.conv.rag_id))

    async def test_a_url_is_fetched_by_the_server(
        self, rag: Rag, sessions: SessionRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = BytesIO(b"<html><body><p>The engine burns fuel.</p></body></html>")
        response.headers = Message()  # type: ignore[attr-defined]
        response.headers["Content-Type"] = "text/html"  # type: ignore[attr-defined]
        monkeypatch.setattr(sources.urllib.request, "urlopen", MagicMock(return_value=response))
        session, out = await open_session(sessions, {})

        await Index(rag).execute(session, "https://example.com/engines")

        assert out.requests() == []
        assert out.system_messages() == ["https://example.com/engines: 1 chunks"]
        assert texts(rag, session.conv.rag_id) == ["The engine burns fuel."]

    async def test_a_failing_source_is_reported_and_skipped(self, rag: Rag, sessions: SessionRegistry) -> None:
        files: dict[str, str | Exception] = {"missing.md": Exception("file not found"), "cats.md": CATS}
        session, out = await open_session(sessions, files)

        await Index(rag).execute(session, "missing.md", "image.png", "cats.md")

        messages = out.system_messages()
        assert "file not found" in messages[0]
        assert messages[0].startswith("missing.md: ")
        assert messages[1].startswith("image.png: ")
        assert messages[2:] == ["cats.md: 1 chunks", "Indexed 1/3 sources into 1 chunks (2 skipped)"]
        assert out.paths_read() == ["missing.md", "cats.md"]
        assert texts(rag, session.conv.rag_id) == ["Cats\n\nCats sleep all day."]

    async def test_several_sources_end_with_a_summary(self, rag: Rag, sessions: SessionRegistry) -> None:
        session, out = await open_session(sessions, {"engines.md": ENGINES, "cats.md": CATS})

        await Index(rag).execute(session, "engines.md", "cats.md")

        assert out.system_messages()[-1] == "Indexed 2/2 sources into 2 chunks"

    async def test_a_cancelled_index_stops_before_the_next_source(self, rag: Rag, sessions: SessionRegistry) -> None:
        session, out = await open_session(sessions, {"engines.md": ENGINES, "cats.md": CATS})

        session.start(Index(rag).execute(session, "engines.md", BLOCKED, "cats.md"))
        await asyncio.wait_for(out.reading_blocked.wait(), timeout=5)
        await session.stop()

        assert out.paths_read() == ["engines.md", BLOCKED]
        assert out.system_messages() == ["engines.md: 1 chunks"]
        assert texts(rag, session.conv.rag_id) == ["Engines\n\nThe engine burns fuel."]

    async def test_a_branch_with_no_rag_gets_one_linked(
        self, rag: Rag, sessions: SessionRegistry, db: SqlAlchemy
    ) -> None:
        session, _ = await open_session(sessions, {"cats.md": CATS})
        branch = db.create_branch(model=MODEL)
        assert branch is not None
        session.conv.uuid, session.conv.rag_id = branch, None

        await Index(rag).execute(session, "cats.md")

        assert session.conv.rag_id is not None
        assert db.get_branch_rag_id(branch) == session.conv.rag_id
        assert texts(rag, session.conv.rag_id) == ["Cats\n\nCats sleep all day."]


class TestBuild:
    def test_index_is_registered_only_with_a_rag(self, rag: Rag, sessions: SessionRegistry, db: SqlAlchemy) -> None:
        pm, tools = ProviderManager(SETTINGS.llm), ToolManager()
        with_rag, without = CommandRegistry(), CommandRegistry()
        with_rag.set_commands(build_commands(sessions, with_rag, db, pm, tools, rag=rag))
        without.set_commands(build_commands(sessions, without, db, pm, tools))

        assert "/index" in with_rag.names()
        assert "/index" not in without.names()
