import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sensai_protocol import AiChunk
from sensai_server.app import create_app
from sensai_server.config.settings import DBSettings, LLMSettings, OllamaSettings, PersistentDBSettings, Settings
from sensai_server.llm.persona import PersonaError, PersonaManager, read_persona_file
from sensai_server.llm.provider import provider_manager
from sensai_server.memory.persistent_db import base
from sensai_server.memory.persistent_db.sql_alchemy.core import SqlAlchemy
from sensai_server.session import Session, SessionRegistry
from sensai_server.state import app_state

from tests.server.fakes import FakeAdapter

MODEL = "a"
SLOW = "slow"
CONFIRM = "confirm"
ASK = "ask"
FILE = "file"
PERSONA_PATH = "x.toml"


async def fake_chat(session: Session, prompt: str) -> None:
    """Answer with the prompt in one chunk. A `slow` prompt then streams forever, until cancelled.

    A `confirm` prompt first asks to run a tool, an `ask` prompt asks the user a question, and the answer is the
    chunk sent. A `file` prompt reads the persona file `x.toml` from the client, and the persona's name, or the
    error, is the chunk sent.
    """
    if prompt == FILE:
        try:
            prompt = (await read_persona_file(session.out, PERSONA_PATH)).name
        except PersonaError as err:
            prompt = err.message
    elif prompt == CONFIRM:
        prompt = await session.out.ask_confirmation("web_search", {"query": "paris"})
    elif prompt == ASK:
        prompt = await session.out.ask_user("Which city?") or "(none)"
    await session.out.send_ai_response(AiChunk(content=prompt))
    if prompt == SLOW:
        await asyncio.sleep(3600)
    await session.out.send_ai_response(AiChunk(done=True))


SETTINGS = Settings(llm=LLMSettings(providers={"ollama": OllamaSettings()}))


def db_url(tmp_path: Path) -> str:
    """The URL of a fresh sqlite file in `tmp_path`.

    A file, not `sqlite://`: the test client runs the app in another thread, and an in-memory database is per
    connection.
    """
    return f"sqlite:///{tmp_path / 'test.db'}"


def settings_with_db(tmp_path: Path) -> Settings:
    """`SETTINGS`, with a fresh persistent database in `tmp_path`."""
    return Settings(llm=SETTINGS.llm, db=DBSettings(persistent=PersistentDBSettings(url=db_url(tmp_path))))


@pytest.fixture(autouse=True)
def fresh_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forget the database opened by a previous test: `set_db` keeps the first one of the process."""
    monkeypatch.setattr(base._State, "db", None)  # noqa: SLF001


@pytest.fixture
def persona_manager(tmp_path: Path) -> PersonaManager:
    """A persona manager on a fresh sqlite file."""
    return PersonaManager(SqlAlchemy(db_url(tmp_path)))


@pytest.fixture(autouse=True)
def fake_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the providers from `FakeAdapter`, so that no server test calls a real provider.

    The adapter serves the models `a` and `b`, and its class state is reset for every test.
    """
    monkeypatch.setitem(provider_manager.PROVIDER_ADAPTERS, "ollama", FakeAdapter)
    monkeypatch.setattr(FakeAdapter, "models", [MODEL, "b"])
    monkeypatch.setattr(FakeAdapter, "refused", set())
    monkeypatch.setattr(FakeAdapter, "failing", set())
    monkeypatch.setattr(FakeAdapter, "unreachable", False)


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    return create_app(settings_with_db(tmp_path), chat=fake_chat)


@pytest.fixture
def registry(app: FastAPI) -> SessionRegistry:
    return app_state(app).registry


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
