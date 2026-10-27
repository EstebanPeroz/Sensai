import asyncio
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sensai_protocol import AiChunk
from sensai_server.app import create_app
from sensai_server.models import DEFAULT_MODELS
from sensai_server.session import Session, SessionRegistry

MODEL = DEFAULT_MODELS[0]
SLOW = "slow"


async def fake_chat(session: Session, prompt: str) -> None:
    """Answer with the prompt in one chunk. A `slow` prompt then streams forever, until cancelled."""
    await session.out.send_ai_response(AiChunk(content=prompt))
    if prompt == SLOW:
        await asyncio.sleep(3600)
    await session.out.send_ai_response(AiChunk(done=True))


async def fake_command(session: Session, command: str) -> None:
    await session.out.send_system_message(f"ran {command}")


@pytest.fixture
def app() -> FastAPI:
    return create_app(chat=fake_chat, command=fake_command)


@pytest.fixture
def registry(app: FastAPI) -> SessionRegistry:
    return app.state.registry


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
