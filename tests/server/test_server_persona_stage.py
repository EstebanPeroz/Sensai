from contextlib import aclosing

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
