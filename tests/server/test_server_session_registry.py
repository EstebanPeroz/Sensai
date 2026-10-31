import asyncio
from uuid import uuid4

from sensai_protocol import Error, Session
from sensai_server.llm.provider import ProviderManager
from sensai_server.session import SessionRegistry

from tests.server.conftest import MODEL, SETTINGS
from tests.server.fakes import FakeAdapter, RecordingOutbound


def _registry() -> SessionRegistry:
    return SessionRegistry(ProviderManager(SETTINGS.llm))


class TestSessionRegistry:
    async def test_open_known_model(self) -> None:
        registry, out = _registry(), RecordingOutbound()
        session = await registry.open(out, MODEL)
        assert session is not None
        assert len(registry) == 1
        assert registry.get(session.client_id) is session
        assert out.sent == [Session(client_id=session.client_id, model=MODEL, commands=[], effort=session.effort)]

    async def test_open_untagged_model_served_as_latest(self) -> None:
        FakeAdapter.models = ["qwen3:latest"]
        registry, out = _registry(), RecordingOutbound()
        assert await registry.open(out, "qwen3") is not None
        assert len(registry) == 1

    async def test_open_unknown_model(self) -> None:
        registry, out = _registry(), RecordingOutbound()
        assert await registry.open(out, "nope") is None
        assert len(registry) == 0
        assert len(out.sent) == 1
        error = out.sent[0]
        assert isinstance(error, Error)
        assert "nope" in error.message
        assert "Available models: a, b" in error.message

    async def test_open_without_reachable_provider_lists_none(self) -> None:
        FakeAdapter.unreachable = True
        registry, out = _registry(), RecordingOutbound()
        assert await registry.open(out, MODEL) is None
        error = out.sent[0]
        assert isinstance(error, Error)
        assert "Available models: none" in error.message

    async def test_close_removes_and_cancels(self) -> None:
        registry = _registry()
        session = await registry.open(RecordingOutbound(), MODEL)
        assert session is not None
        started = asyncio.Event()

        async def forever() -> None:
            started.set()
            await asyncio.sleep(3600)

        session.start(forever())
        await started.wait()
        task = session.task
        await registry.close(session.client_id)
        assert len(registry) == 0
        assert registry.get(session.client_id) is None
        assert task is not None
        assert task.cancelled()

    async def test_close_unknown_id(self) -> None:
        registry = _registry()
        await registry.open(RecordingOutbound(), MODEL)
        await registry.close(uuid4())
        assert len(registry) == 1

    async def test_two_sessions_get_different_ids(self) -> None:
        registry = _registry()
        first = await registry.open(RecordingOutbound(), MODEL)
        second = await registry.open(RecordingOutbound(), MODEL)
        assert first is not None
        assert second is not None
        assert first.client_id != second.client_id
        assert len(registry) == 2
