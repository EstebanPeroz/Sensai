from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable

    from sensai_server.llm.persona import Persona
    from sensai_server.llm.responses import ChatResponse
    from sensai_server.pipeline.chain import ChatContext, Next


class PersonaStage:
    """Prepend the security personas, then the conversation's persona, to the payload, then let the chain go on.

    The security personas are the same for every chat, and come first so that no loaded persona overrides them. The
    conversation's persona comes from the context, a snapshot of the conversation's own. Neither is ever written to
    the history.
    """

    _security: list[Persona]

    def __init__(self, security: Iterable[Persona] = ()) -> None:
        """Prepend the prompts of the `security` personas, in their order, to every chat."""
        self._security = list(security)

    async def __call__(self, context: ChatContext, next_stage: Next) -> AsyncGenerator[ChatResponse]:
        """Add the security personas and the persona to `context`, and stream what the rest of the chain answers.

        A loaded persona stored as one of the security personas is only sent once.
        """
        personas = list(self._security)
        persona = context.persona
        if persona is not None and (persona.uuid is None or all(persona.uuid != guard.uuid for guard in personas)):
            personas.append(persona)
        if personas:
            context.messages = [*(each.to_message().to_dict() for each in personas), *context.messages]
        async with aclosing(next_stage(context)) as chunks:
            async for chunk in chunks:
                yield chunk
