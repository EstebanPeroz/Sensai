from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING

from sensai_protocol import AiChunk
from sensai_protocol.server_messages import Role

from sensai_server.chat.react import EFFORTS
from sensai_server.pipeline import ChatContext

if TYPE_CHECKING:
    from sensai_server.app.outbound import Outbound
    from sensai_server.llm.responses import ChatResponse
    from sensai_server.memory.persistent_db import PersistentDatabase
    from sensai_server.pipeline import ChatPipeline
    from sensai_server.session import Session


async def run_chat(session: Session, prompt: str, pipeline: ChatPipeline, db: PersistentDatabase) -> None:
    """Run the chat pipeline."""
    conv = session.conv
    if not await conv.save_message(Role.USER, prompt, session.out, db):
        return
    context = ChatContext(
        message=prompt,
        out=session.out,
        model=str(conv.model),
        persona=conv.persona,
        messages=conv.history.to_json(),
        tools=session.tools,
        max_iterations=EFFORTS[session.effort],
        rag_id=conv.rag_id,
        cache=session.cache,
    )
    content = await _stream(session.out, pipeline, context)
    if content.strip() == "":
        return
    await conv.save_message(Role.ASSISTANT, content, session.out, db)


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
