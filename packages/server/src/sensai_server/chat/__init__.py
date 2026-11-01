from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from sensai_protocol import CompressionChoice
from sensai_protocol.error import SensaiError

from sensai_server.cache import CacheError
from sensai_server.chat.chat import run_chat
from sensai_server.chat.usage import send_context_usage

if TYPE_CHECKING:
    from sensai_server.events import EventBus
    from sensai_server.memory.persistent_db import PersistentDatabase
    from sensai_server.monitoring.budget import ContextBudget
    from sensai_server.pipeline import ChatPipeline
    from sensai_server.rag import Rag
    from sensai_server.session import Session


BRANCH_NAME_LENGTH = 50
"""The length a branch name, taken from the first prompt, is cut to, so it shows well in `/branch list`."""
_NO_MODEL = "No model is set for this conversation."


async def chat(  # noqa: PLR0913, PLR0917 - the shared services a chat needs come through here
    session: Session,
    prompt: str,
    pipeline: ChatPipeline,
    db: PersistentDatabase,
    events: EventBus,
    rag: Rag | None = None,
    *,
    budget: ContextBudget | None = None,
) -> None:
    """Send `prompt` through `pipeline` and stream the answer to the session as `ai_chunk`.

    The prompt is not echoed: `handle_chat` already did. The branch is created in `db` on the first message, with
    its collection in `rag`, if any. The prompt and the answer are both saved. Every call to the model is emitted on
    `events`, and the new context usage from `budget`, if any, is sent once the answer is saved. When the branch is
    near its budget, the client is first asked whether to compress it or to start a new conversation. A `SensaiError` is
    sent as an `error`. A cancellation saves nothing more and propagates, closing the pipeline down to the model's
    stream.
    """
    try:
        await _ask_compression(session, budget)
        await _set_chat(session, prompt, db, rag)
        await run_chat(session, prompt, pipeline, db, events, budget=budget)
    except SensaiError as err:
        await session.out.send_error(str(err))


async def _ask_compression(session: Session, budget: ContextBudget | None) -> None:
    """Ask how to go on when the branch is near its budget, before the prompt is saved.

    On `new_conversation`, the session leaves its branch, so that the prompt starts a new one. On `compress`, the
    pipeline's compression stage summarises the branch.
    """
    conv = session.conv
    if budget is None or conv.model is None or conv.uuid is None:
        return
    usage = await budget.usage(conv.model, conv.uuid)
    if not usage.near_limit:
        return
    if await session.out.ask_compression(usage) is CompressionChoice.NEW_CONVERSATION:
        await _new_conversation(session, budget)


async def _new_conversation(session: Session, budget: ContextBudget) -> None:
    """Leave the session's branch, clear its cached answers, and send the client the empty conversation."""
    session.conv.reset()
    if session.cache is not None:
        try:
            await session.cache.clear()
        except CacheError as error:
            await session.out.send_system_message(f"The semantic cache could not be cleared: {error}")
    await session.out.load_conversation([])
    await send_context_usage(session, budget)


async def _set_chat(session: Session, prompt: str, db: PersistentDatabase, rag: Rag | None = None) -> None:

    conv = session.conv
    if conv.model is None:
        raise SensaiError(_NO_MODEL)
    if conv.uuid is not None:
        return
    await asyncio.to_thread(conv.add_conv_to_db, db, None, prompt[:BRANCH_NAME_LENGTH])
    if rag is not None and conv.uuid is not None and conv.rag_id is not None:
        try:
            await rag.create(conv.uuid, conv.rag_id)
        except SensaiError as err:
            await session.out.send_system_message(f"The RAG of this conversation could not be created: {err}")
