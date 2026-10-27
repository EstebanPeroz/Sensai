import asyncio
from uuid import uuid4

from sensai_protocol import Error
from sensai_server.session import DEFAULT_EFFORT, Session

from tests.server.fakes import RecordingOutbound


def _session() -> tuple[Session, RecordingOutbound]:
    out = RecordingOutbound()
    return Session(uuid4(), "fake-model", out), out


async def _forever(started: asyncio.Event) -> None:
    started.set()
    await asyncio.sleep(3600)


class TestSession:
    def test_defaults(self) -> None:
        session, _ = _session()
        assert session.effort == DEFAULT_EFFORT
        assert session.task is None
        assert not session.busy

    async def test_start_refuses_second_task(self) -> None:
        session, _ = _session()
        started = asyncio.Event()
        assert session.start(_forever(started))
        await started.wait()
        second = _forever(asyncio.Event())
        assert not session.start(second)
        assert second.cr_frame is None  # closed, never run
        await session.close()

    async def test_task_reset_when_done(self) -> None:
        session, out = _session()

        async def quick() -> None:
            await out.send_error("ran")

        assert session.start(quick())
        assert session.task is not None
        await session.task
        assert session.task is None
        assert not session.busy
        assert out.sent == [Error(message="ran")]

    async def test_cancel_stops_task_silently(self) -> None:
        session, out = _session()
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

    def test_cancel_without_task_does_nothing(self) -> None:
        session, _ = _session()
        session.cancel()

    async def test_exception_becomes_error(self) -> None:
        session, out = _session()

        async def boom() -> None:
            msg = "provider down"
            raise RuntimeError(msg)

        session.start(boom())
        assert session.task is not None
        await asyncio.wait({session.task})
        assert out.sent == [Error(message="provider down")]
        assert not session.busy

    async def test_close_waits_for_task(self) -> None:
        session, out = _session()
        started = asyncio.Event()
        session.start(_forever(started))
        await started.wait()
        task = session.task
        await session.close()
        assert task is not None
        assert task.done()
        assert not session.busy
        assert out.sent == []

    async def test_close_without_task(self) -> None:
        session, _ = _session()
        await session.close()
