from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sensai.llm.responses import ChatResponse
    from sensai.llm.thinking.react import ReAct
    from sensai.pipeline.chain import ChatContext


class ModelStage:
    """Hand the payload the earlier stages built to the ReAct loop, and stream what it answers."""

    def __init__(self, react: ReAct) -> None:
        """Keep the ReAct loop this stage asks for an answer."""
        self._react = react

    def __call__(self, context: ChatContext) -> Iterator[ChatResponse]:
        """Stream the answer to the payload in `context`.

        Takes no `next_stage`: nothing runs after the model, it closes the chain.
        """
        return self._react.generate_answer(context.messages, context.on_interaction)
