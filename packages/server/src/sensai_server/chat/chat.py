from __future__ import annotations

import asyncio
import logging
from contextlib import aclosing
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

from sensai_protocol import AiChunk
from sensai_protocol.error import SensaiError
from sensai_protocol.server_messages import Role

from sensai_server.chat.react import EFFORTS
from sensai_server.pipeline import ChatContext

if TYPE_CHECKING:
    from uuid import UUID

    from sensai_server.app.outbound import Outbound
    from sensai_server.events import EventBus
    from sensai_server.llm.responses import ChatResponse
    from sensai_server.memory.persistent_db import PersistentDatabase
    from sensai_server.monitoring.interaction import Interaction
    from sensai_server.pipeline import ChatPipeline
    from sensai_server.session import Session

BRANCH_NAME_LENGTH = 50
"""The length a branch name, taken from the first prompt, is cut to, so it shows well in `/branch list`."""
_NO_MODEL = "No model is set for this conversation."

logger = logging.getLogger(__name__)


async def run_chat(
    session: Session, prompt: str, pipeline: ChatPipeline, db: PersistentDatabase, events: EventBus
) -> None:
    """Run the chat pipeline, emitting every call it makes to the model on `events`, tagged with the branch."""
    conv = session.conv
    if conv.model is None:
        raise SensaiError(_NO_MODEL)
    if conv.uuid is None:
        await asyncio.to_thread(conv.add_conv_to_db, db, None, prompt[:BRANCH_NAME_LENGTH])
    # TODO(cache): answer from the semantic cache here, as an outer stage right after the guard.  # noqa: TD003, FIX002
    if not await conv.save_message(Role.USER, prompt, session.out, db):
        return
    context = ChatContext(
        message=prompt,
        out=session.out,
        model=conv.model,
        persona=conv.persona,
        messages=conv.history.to_json(),
        tools=session.tools,
        max_iterations=EFFORTS[session.effort],
        on_interaction=partial(
            _emit_interaction, events, conv.uuid, conv.persona.name if conv.persona is not None else None
        ),
    )
    content = await _stream(session.out, pipeline, context)
    if content.strip() == "":
        return
    await conv.save_message(Role.ASSISTANT, content, session.out, db)
    # TODO(cache): store the complete answer in the semantic cache.  # noqa: TD003, FIX002


def _emit_interaction(events: EventBus, branch_id: UUID | None, persona: str | None, interaction: Interaction) -> None:
    """Tag `interaction` with its `branch_id` and `persona`, then emit it on `events`.

    It runs in the `finally` of a model call that may be closing, so it never raises: a failing subscriber is only
    logged, and the answer goes on.
    """
    try:
        events.emit(replace(interaction, branch_id=branch_id, persona=persona))
    except Exception:
        logger.exception("Failed to emit an interaction")


async def _stream(out: Outbound, pipeline: ChatPipeline, context: ChatContext) -> str:
    """Send the pipeline's chunks to `out` until they run out or one has an error, and return the answer's content."""
    content = ""
    async with aclosing(pipeline.execute(context)) as chunks:
        async for chunk in chunks:
            if chunk.content is not None:
                content += chunk.content
            await out.send_ai_response(_to_ai_chunk(chunk))
            if chunk.error is not None:
                break
    return content


def _to_ai_chunk(response: ChatResponse) -> AiChunk:
    """Turn a pipeline chunk into its wire format."""
    return AiChunk(
        content=response.content,
        thinking=response.thinking,
        tool_calls=response.tool_calls or None,
        done=response.done,
        error=response.error,
    )
