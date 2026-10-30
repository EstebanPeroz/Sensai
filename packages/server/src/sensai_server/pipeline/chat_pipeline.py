from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sensai_server.llm.responses import ChatResponse
    from sensai_server.pipeline.chain import ChatContext, Next, Stage


class ChatPipeline:
    """Run an exchange through an ordered chain of stages wrapped around the model.

    Each stage is handed the rest of the chain, so it may answer in place of
    the model by never calling it, enrich the context before calling it, or
    stay on the line and work on the chunks it streams back.

    It holds no state per exchange, so one instance is shared by every session.
    """

    def __init__(self, model: Next) -> None:
        """Start with a chain made of `model` alone, the step that actually answers."""
        self._stages: list[Stage] = []
        self._model = model

    def use(self, stage: Stage) -> ChatPipeline:
        """Wrap `stage` around the chain built so far, and return the pipeline to allow chaining."""
        self._stages.append(stage)
        return self

    def execute(self, context: ChatContext) -> AsyncIterator[ChatResponse]:
        """Stream the answer to `context`, through every stage and back."""
        return self._chain()(context)

    def _chain(self) -> Next:
        """Fold the stages into the model, from the last one to the first."""
        chain = self._model
        for stage in reversed(self._stages):
            chain = self._link(stage, chain)
        return chain

    @staticmethod
    def _link(stage: Stage, next_stage: Next) -> Next:
        """Bind `stage` to the `next_stage` it precedes, making the pair one step."""

        def linked(context: ChatContext) -> AsyncIterator[ChatResponse]:
            return stage(context, next_stage)

        return linked
