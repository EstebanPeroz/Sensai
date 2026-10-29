import asyncio
from uuid import uuid4

from sensai_protocol import Error, Session
from sensai_server.llm.persona import Persona, PersonaManager
from sensai_server.llm.provider import ProviderManager
from sensai_server.session import SessionRegistry

from tests.server.conftest import MODEL, SETTINGS
from tests.server.fakes import FakeAdapter, RecordingOutbound


def _registry(persona_manager: PersonaManager) -> SessionRegistry:
    return SessionRegistry(ProviderManager(SETTINGS.llm), persona_manager)


class TestSessionRegistry:
    async def test_open_known_model(self, persona_manager: PersonaManager) -> None:
        registry, out = _registry(persona_manager), RecordingOutbound()
        session = await registry.open(out, MODEL)
        assert session is not None
        assert len(registry) == 1
        assert registry.get(session.client_id) is session
        assert out.sent == [Session(client_id=session.client_id, model=MODEL, commands=[], effort=session.effort)]

    async def test_open_untagged_model_served_as_latest(self, persona_manager: PersonaManager) -> None:
        FakeAdapter.models = ["qwen3:latest"]
        registry, out = _registry(persona_manager), RecordingOutbound()
        assert await registry.open(out, "qwen3") is not None
        assert len(registry) == 1

    async def test_open_unknown_model(self, persona_manager: PersonaManager) -> None:
        registry, out = _registry(persona_manager), RecordingOutbound()
        assert await registry.open(out, "nope") is None
        assert len(registry) == 0
        assert len(out.sent) == 1
        error = out.sent[0]
        assert isinstance(error, Error)
        assert "nope" in error.message
        assert "Available models: a, b" in error.message

    async def test_open_without_reachable_provider_lists_none(self, persona_manager: PersonaManager) -> None:
        FakeAdapter.unreachable = True
        registry, out = _registry(persona_manager), RecordingOutbound()
        assert await registry.open(out, MODEL) is None
        error = out.sent[0]
        assert isinstance(error, Error)
        assert "Available models: none" in error.message

    async def test_close_removes_and_cancels(self, persona_manager: PersonaManager) -> None:
        registry = _registry(persona_manager)
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

    async def test_close_unknown_id(self, persona_manager: PersonaManager) -> None:
        registry = _registry(persona_manager)
        await registry.open(RecordingOutbound(), MODEL)
        await registry.close(uuid4())
        assert len(registry) == 1

    async def test_two_sessions_get_different_ids(self, persona_manager: PersonaManager) -> None:
        registry = _registry(persona_manager)
        first = await registry.open(RecordingOutbound(), MODEL)
        second = await registry.open(RecordingOutbound(), MODEL)
        assert first is not None
        assert second is not None
        assert first.client_id != second.client_id
        assert len(registry) == 2

    async def test_sessions_share_the_persona_manager(self, persona_manager: PersonaManager) -> None:
        registry = _registry(persona_manager)
        first = await registry.open(RecordingOutbound(), MODEL)
        second = await registry.open(RecordingOutbound(), MODEL)
        assert first is not None
        assert second is not None
        assert first.persona_manager is persona_manager
        assert second.persona_manager is persona_manager


class TestRefreshPersona:
    async def test_a_new_session_has_no_persona(self, persona_manager: PersonaManager) -> None:
        session = await _registry(persona_manager).open(RecordingOutbound(), MODEL)
        assert session is not None
        assert session.conv.persona is None

    async def test_loading_in_one_session_leaves_the_other(self, persona_manager: PersonaManager) -> None:
        registry = _registry(persona_manager)
        first = await registry.open(RecordingOutbound(), MODEL)
        second = await registry.open(RecordingOutbound(), MODEL)
        assert first is not None
        assert second is not None
        first.conv.persona = persona_manager.create_default_persona()
        assert second.conv.persona is None

    async def test_update_refreshes_every_session_using_it(self, persona_manager: PersonaManager) -> None:
        registry = _registry(persona_manager)
        first = await registry.open(RecordingOutbound(), MODEL)
        second = await registry.open(RecordingOutbound(), MODEL)
        assert first is not None
        assert second is not None
        persona = persona_manager.create_default_persona()
        first.conv.persona = second.conv.persona = persona
        updated = persona_manager.update_persona(persona.name, prompt="new prompt")
        registry.refresh_persona(updated)
        assert first.conv.persona == updated
        assert second.conv.persona == updated
        assert updated.prompt == "new prompt"

    async def test_sessions_on_another_persona_or_none_are_left(self, persona_manager: PersonaManager) -> None:
        registry = _registry(persona_manager)
        using_other = await registry.open(RecordingOutbound(), MODEL)
        using_none = await registry.open(RecordingOutbound(), MODEL)
        assert using_other is not None
        assert using_none is not None
        persona = persona_manager.create_default_persona()
        other = persona_manager.create_persona(Persona(None, "Other", "another one", "be other"))
        using_other.conv.persona = other
        registry.refresh_persona(persona_manager.update_persona(persona.name, prompt="new prompt"))
        assert using_other.conv.persona == other
        assert using_none.conv.persona is None
