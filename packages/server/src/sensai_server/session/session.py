from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sensai_server.cache import CacheError
from sensai_server.memory import Conversation

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from uuid import UUID

    from sensai_server.app import Outbound
    from sensai_server.cache import SemanticCache, SessionCache
    from sensai_server.llm.persona import PersonaManager
    from sensai_server.llm.provider.provider_manager import ProviderManager
    from sensai_server.tools.state import ToolState
    from sensai_server.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)

DEFAULT_EFFORT = "high"


class Session:
    """The state of one logged-in client, and the single task it may run at a time."""

    cache: SessionCache | None
    client_id: UUID
    conv: Conversation
    effort: str
    out: Outbound
    persona_manager: PersonaManager
    provider_manager: ProviderManager
    task: asyncio.Task[None] | None
    task_label: str | None
    tools: ToolState

    def __init__(  # noqa: PLR0913 - the shared managers, plus the keyword-only options
        self,
        out: Outbound,
        pm: ProviderManager,
        persona_manager: PersonaManager,
        tool_manager: ToolManager,
        *,
        cache: SemanticCache | None = None,
        effort: str = DEFAULT_EFFORT,
    ) -> None:
        """Build the session, sending on `out`, using the shared managers.

        The session gets its own tool state from `tool_manager`, so its tool choices never reach the other sessions,
        and its own view of `cache`, if there is one, so it never sees another session's cached answers.
        """
        self.client_id = uuid4()
        self.effort = effort
        self.out = out
        self.persona_manager = persona_manager
        self.task = None
        self.task_label = None
        self.provider_manager = pm
        self.conv = Conversation()
        self.tools = tool_manager.new_state()
        self.cache = cache.for_session(self.client_id) if cache is not None else None

    async def set_model(self, model: str) -> bool:
        """Set the model the Session will use from now on."""
        provider = self.provider_manager.get_provider(model)
        if provider is None:
            return False
        self.conv.model = model
        return True

    @property
    def busy(self) -> bool:
        """Whether a task is running."""
        return self.task is not None and not self.task.done()

    def start(self, coro: Coroutine[Any, Any, None], label: str | None = None) -> bool:
        """Run `coro` as the session's task. Returns False, without running it, if a task is already running.

        `label` names the task in messages, such as the typed command; it is kept in `task_label` while it runs.

        An exception raised by `coro` is sent to the client as an `error`; a cancellation sends nothing.
        """
        if self.busy:
            coro.close()
            return False
        self.task = asyncio.create_task(self._run(coro))
        self.task_label = label
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
                self.task_label = None

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
        """Release the session: stop its running task, then clear its cached answers.

        A failure to clear the cache is logged, never raised: closing always succeeds.
        """
        await self.stop()
        if self.cache is None:
            return
        try:
            await self.cache.clear()
        except CacheError as error:
            logger.warning("the semantic cache of session %s could not be cleared: %s", self.client_id, error)
