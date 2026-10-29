import asyncio
from uuid import uuid4

from sensai_protocol import Error
from sensai_server.llm.persona import PersonaManager
from sensai_server.llm.provider import ProviderManager
from sensai_server.session.session import DEFAULT_EFFORT, Session

from tests.server.conftest import SETTINGS
from tests.server.fakes import RecordingOutbound


def _session(persona_manager: PersonaManager) -> tuple[Session, RecordingOutbound]:
    out = RecordingOutbound()
    return Session(uuid4(), out, ProviderManager(SETTINGS.llm), persona_manager), out


async def _forever(started: asyncio.Event) -> None:
    started.set()
    await asyncio.sleep(3600)


class TestSession:
    def test_defaults(self, persona_manager: PersonaManager) -> None:
        session, _ = _session(persona_manager)
        assert session.effort == DEFAULT_EFFORT
        assert session.task is None
        assert not session.busy

    async def test_start_refuses_second_task(self, persona_manager: PersonaManager) -> None:
        session, _ = _session(persona_manager)
        started = asyncio.Event()
        assert session.start(_forever(started))
        await started.wait()
        second = _forever(asyncio.Event())
        assert not session.start(second)
        assert second.cr_frame is None  # closed, never run
        await session.close()

    async def test_task_reset_when_done(self, persona_manager: PersonaManager) -> None:
        session, out = _session(persona_manager)

        async def quick() -> None:
            await out.send_error("ran")

        assert session.start(quick())
        assert session.task is not None
        await session.task
        assert session.task is None
        assert not session.busy
        assert out.sent == [Error(message="ran")]

    async def test_cancel_stops_task_silently(self, persona_manager: PersonaManager) -> None:
        session, out = _session(persona_manager)
        started = asyncio.Event()
        session.start(_forever(started))
        await started.wait()
        task = session.task
        assert task is not None
        session.cancel()
        await asyncio.wait({task})
        assert task.cancelled()
        assert session.task is None
        assert out.sent == []

    def test_cancel_without_task_does_nothing(self, persona_manager: PersonaManager) -> None:
        session, _ = _session(persona_manager)
        session.cancel()

    async def test_exception_becomes_error(self, persona_manager: PersonaManager) -> None:
        session, out = _session(persona_manager)

        async def boom() -> None:
            msg = "provider down"
            raise RuntimeError(msg)

        session.start(boom())
        assert session.task is not None
        await asyncio.wait({session.task})
        assert out.sent == [Error(message="provider down")]
        assert not session.busy

    async def test_close_waits_for_task(self, persona_manager: PersonaManager) -> None:
        session, out = _session(persona_manager)
        started = asyncio.Event()
        session.start(_forever(started))
        await started.wait()
        task = session.task
        await session.close()
        assert task is not None
        assert task.done()
        assert not session.busy
        assert out.sent == []

    async def test_close_without_task(self, persona_manager: PersonaManager) -> None:
        session, _ = _session(persona_manager)
        await session.close()
