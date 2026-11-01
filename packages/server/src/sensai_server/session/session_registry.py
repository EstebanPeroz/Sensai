from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from sensai_protocol import Session as SessionMessage

from sensai_server.session import Session

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sensai_server.app import Outbound
    from sensai_server.cache import SemanticCache
    from sensai_server.llm.persona import Persona, PersonaManager
    from sensai_server.llm.provider import ProviderManager
    from sensai_server.tools.tool_manager import ToolManager


class SessionRegistry:
    """The live sessions, by client id: the connection list."""

    _sessions: dict[UUID, Session]
    _lock: asyncio.Lock
    _cache: SemanticCache | None
    _pm: ProviderManager
    _persona_manager: PersonaManager
    _tool_manager: ToolManager

    def __init__(
        self,
        pm: ProviderManager,
        persona_manager: PersonaManager,
        tool_manager: ToolManager,
        cache: SemanticCache | None = None,
    ) -> None:
        """Open sessions on the models served by `pm`, sharing `persona_manager`, `tool_manager` and `cache`.

        Each session gets its own view of `cache`; without one, the sessions have no semantic cache.
        """
        self._pm = pm
        self._cache = cache
        self._persona_manager = persona_manager
        self._tool_manager = tool_manager
        self._sessions = {}
        self._lock = asyncio.Lock()

    async def open(self, out: Outbound, model: str, commands: Sequence[str] = ()) -> Session | None:
        """Open and register a session on `model`, and answer the client with `session`, listing `commands`.

        If no provider serves `model`, an `error` listing the available models is sent and None is returned.
        """
        provider = self._pm.get_provider(model)
        if provider is None:
            await out.send_error(
                f"Unknown model '{model}'. Available models: {', '.join(self._pm.get_models()) or 'none'}"
            )
            return None
        session = Session(out, self._pm, self._persona_manager, self._tool_manager, cache=self._cache)
        await session.set_model(model)
        async with self._lock:
            self._sessions[session.client_id] = session
        await out.send(
            SessionMessage(client_id=session.client_id, model=model, commands=list(commands), effort=session.effort)
        )
        return session

    async def close(self, client_id: UUID) -> None:
        """Stop the task of the session of `client_id`, clear its cache and unregister it. An unknown id is ignored."""
        async with self._lock:
            session = self._sessions.pop(client_id, None)
        if session is not None:
            await session.close()

    async def close_all(self) -> None:
        """Stop the task of every session and unregister them all, as the server shuts down."""
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        await asyncio.gather(*(session.close() for session in sessions))

    def refresh_persona(self, updated: Persona) -> None:
        """Give `updated` to every session whose conversation uses an older version of that persona."""
        for session in self._sessions.values():
            persona = session.conv.persona
            if persona is not None and persona.uuid == updated.uuid:
                session.conv.persona = updated

    def get(self, client_id: UUID) -> Session | None:
        """Return the session of `client_id`, if it is open."""
        return self._sessions.get(client_id)

    def __len__(self) -> int:
        """Return the number of open sessions."""
        return len(self._sessions)
