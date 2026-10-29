from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from sensai_server.memory import Conversation

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from uuid import UUID

    from sensai_server.llm.provider.provider_manager import ProviderManager
    from sensai_server.outbound import Outbound

DEFAULT_EFFORT = "high"


class Session:
    """The state of one logged-in client, and the single task it may run at a time."""

    client_id: UUID
    conv: Conversation
    effort: str
    out: Outbound
    task: asyncio.Task[None] | None

    def __init__(
        self, client_id: UUID, model: str, out: Outbound, pm: ProviderManager, effort: str = DEFAULT_EFFORT
    ) -> None:
        """Build the session of `client_id`, chatting with `model` and sending on `out`."""
        self.client_id = client_id
        self.model = model
        self.effort = effort
        self.out = out
        self.task = None
        self.conv = Conversation(provider_manager=pm, model=model)

    @property
    def busy(self) -> bool:
        """Whether a task is running."""
        return self.task is not None and not self.task.done()

    def start(self, coro: Coroutine[Any, Any, None]) -> bool:
        """Run `coro` as the session's task. Returns False, without running it, if a task is already running.

        An exception raised by `coro` is sent to the client as an `error`; a cancellation sends nothing.
        """
        if self.busy:
            coro.close()
            return False
        self.task = asyncio.create_task(self._run(coro))
        return True

    async def _run(self, coro: Coroutine[Any, Any, None]) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001 - any failure of the task is reported to the client
            await self.out.send_error(str(err) or type(err).__name__)
        finally:
            if self.task is asyncio.current_task():
                self.task = None

    def cancel(self) -> None:
        """Cancel the running task, if any."""
        if self.task is not None:
            self.task.cancel()

    async def stop(self) -> None:
        """Cancel the running task, if any, and wait until it has stopped."""
        task = self.task
        if task is None:
            return
        task.cancel()
        await asyncio.wait({task})

    async def close(self) -> None:
        """Release the session: stop its running task."""
        await self.stop()
