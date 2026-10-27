"""Put the active persona in front of the payload before the model reads it."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sensai.llm.persona_manager import PersonaManager
    from sensai.llm.responses import ChatResponse
    from sensai.pipeline.chain import ChatContext, Next


class PersonaStage:
    """Prepend the active persona to the payload, then let the chain go on."""

    def __init__(self, persona_manager: PersonaManager) -> None:
        """Keep the manager this stage reads the active persona from."""
        self._persona_manager = persona_manager

    def __call__(self, context: ChatContext, next_stage: Next) -> Iterator[ChatResponse]:
        """Add the persona to `context` when there is one, and stream what the rest of the chain answers."""
        persona = self._persona_manager.active_persona()
        if persona is not None:
            context.messages = [persona.to_payload(), *context.messages]
        return next_stage(context)
