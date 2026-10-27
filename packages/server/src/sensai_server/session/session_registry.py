from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import uuid4

from sensai_protocol import Session as SessionMessage

from sensai_server.session import Session

if TYPE_CHECKING:
    from uuid import UUID

    from sensai_server.models import ModelCatalog
    from sensai_server.outbound import Outbound


class SessionRegistry:
    """The live sessions, by client id: the connection list."""

    _catalog: ModelCatalog
    _sessions: dict[UUID, Session]
    _lock: asyncio.Lock

    def __init__(self, catalog: ModelCatalog) -> None:
        """Open sessions on the models of `catalog`."""
        self._catalog = catalog
        self._sessions = {}
        self._lock = asyncio.Lock()

    async def open(self, out: Outbound, model: str) -> Session | None:
        """Open and register a session on `model`, and answer the client with `session`.

        If no provider serves `model`, an `error` listing the available models is sent and None is returned.
        """
        models = await self._catalog.available_models()
        if model not in models:
            await out.send_error(f"Unknown model '{model}'. Available models: {', '.join(models) or 'none'}")
            return None
        session = Session(uuid4(), model, out)
        async with self._lock:
            self._sessions[session.client_id] = session
        await out.send(
            SessionMessage(client_id=session.client_id, model=session.model, commands=[], effort=session.effort)
        )
        return session

    async def close(self, client_id: UUID) -> None:
        """Stop the task of the session of `client_id` and unregister it. An unknown id does nothing."""
        async with self._lock:
            session = self._sessions.pop(client_id, None)
        if session is not None:
            await session.close()

    def get(self, client_id: UUID) -> Session | None:
        """Return the session of `client_id`, if it is open."""
        return self._sessions.get(client_id)

    def __len__(self) -> int:
        """Return the number of open sessions."""
        return len(self._sessions)
