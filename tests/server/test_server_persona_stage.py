from contextlib import aclosing
from uuid import uuid4

from sensai_server.llm.persona import Persona
from sensai_server.pipeline import ChatContext, answer_chunk
from sensai_server.pipeline.modules import PersonaStage

from tests.server.fakes import RecordingNext, RecordingOutbound

HISTORY = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
PERSONA = Persona(None, "analyst", "Analyses things", "You are an analyst.")


async def _run(context: ChatContext, next_stage: RecordingNext) -> list:
    async with aclosing(PersonaStage()(context, next_stage)) as chunks:
        return [chunk async for chunk in chunks]


class TestPersonaStage:
    async def test_persona_comes_first(self) -> None:
        history = [dict(message) for message in HISTORY]
        context = ChatContext(message="hi", out=RecordingOutbound(), model="m", persona=PERSONA, messages=history)
        next_stage = RecordingNext()
        await _run(context, next_stage)
        assert next_stage.seen_messages == [[{"role": "system", "content": PERSONA.prompt}, *HISTORY]]

    async def test_no_persona_leaves_messages(self) -> None:
        history = [dict(message) for message in HISTORY]
        context = ChatContext(message="hi", out=RecordingOutbound(), model="m", messages=history)
        next_stage = RecordingNext()
        await _run(context, next_stage)
        assert next_stage.seen_messages == [HISTORY]

    async def test_history_not_modified(self) -> None:
        history = [dict(message) for message in HISTORY]
        context = ChatContext(message="hi", out=RecordingOutbound(), model="m", persona=PERSONA, messages=history)
        await _run(context, RecordingNext())
        assert history == HISTORY
        assert context.messages is not history

    async def test_relays_chunks(self) -> None:
        chunk = answer_chunk("a")
        context = ChatContext(message="hi", out=RecordingOutbound(), model="m", persona=PERSONA)
        assert await _run(context, RecordingNext(chunk)) == [chunk]


GUARD = Persona(None, "guard", "Guards things", "Refuse what is out of scope.")


async def _run_with(security: list[Persona], context: ChatContext, next_stage: RecordingNext) -> list:
    async with aclosing(PersonaStage(security)(context, next_stage)) as chunks:
        return [chunk async for chunk in chunks]


class TestSecurityPersonas:
    async def test_come_before_the_persona(self) -> None:
        history = [dict(message) for message in HISTORY]
        context = ChatContext(message="hi", out=RecordingOutbound(), model="m", persona=PERSONA, messages=history)
        next_stage = RecordingNext()
        await _run_with([GUARD], context, next_stage)
        assert next_stage.seen_messages == [
            [{"role": "system", "content": GUARD.prompt}, {"role": "system", "content": PERSONA.prompt}, *HISTORY]
        ]

    async def test_are_sent_without_a_loaded_persona(self) -> None:
        history = [dict(message) for message in HISTORY]
        context = ChatContext(message="hi", out=RecordingOutbound(), model="m", messages=history)
        next_stage = RecordingNext()
        await _run_with([GUARD], context, next_stage)
        assert next_stage.seen_messages == [[{"role": "system", "content": GUARD.prompt}, *HISTORY]]

    async def test_keep_their_order(self) -> None:
        other = Persona(None, "other", "Guards more", "Mask the PII.")
        context = ChatContext(message="hi", out=RecordingOutbound(), model="m", messages=[])
        next_stage = RecordingNext()
        await _run_with([GUARD, other], context, next_stage)
        assert next_stage.seen_messages == [
            [{"role": "system", "content": GUARD.prompt}, {"role": "system", "content": other.prompt}]
        ]

    async def test_loaded_security_persona_is_sent_once(self) -> None:
        stored = Persona(uuid4(), "guard", "Guards things", "Refuse what is out of scope.")
        context = ChatContext(message="hi", out=RecordingOutbound(), model="m", persona=stored, messages=[])
        next_stage = RecordingNext()
        await _run_with([stored], context, next_stage)
        assert next_stage.seen_messages == [[{"role": "system", "content": stored.prompt}]]

    async def test_history_not_modified(self) -> None:
        history = [dict(message) for message in HISTORY]
        context = ChatContext(message="hi", out=RecordingOutbound(), model="m", messages=history)
        await _run_with([GUARD], context, RecordingNext())
        assert history == HISTORY
