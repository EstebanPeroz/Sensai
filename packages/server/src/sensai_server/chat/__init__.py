from __future__ import annotations

from typing import TYPE_CHECKING

from sensai_protocol.error import SensaiError

from sensai_server.chat.chat import run_chat

if TYPE_CHECKING:
    from sensai_server.memory.persistent_db import PersistentDatabase
    from sensai_server.pipeline import ChatPipeline
    from sensai_server.session import Session


async def chat(session: Session, prompt: str, pipeline: ChatPipeline, db: PersistentDatabase) -> None:
    """Send `prompt` through `pipeline` and stream the answer to the session as `ai_chunk`.

    The prompt is not echoed: `handle_chat` already did. The branch is created in `db` on the first message, and
    the prompt and the answer are both saved. A `SensaiError` is sent as an `error`. A cancellation saves nothing
    more and propagates, closing the pipeline down to the model's stream.
    """
    try:
        await run_chat(session, prompt, pipeline, db)
    except SensaiError as err:
        await session.out.send_error(str(err))
