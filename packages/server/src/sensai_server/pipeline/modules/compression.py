from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING

from sensai_protocol.error import SensaiError
from sensai_protocol.server_messages import Role

from sensai_server.llm.responses import ChatMetrics
from sensai_server.llm.threaded import iterate_in_thread
from sensai_server.memory.summary import Summary
from sensai_server.monitoring.interaction import InteractionStatus, ignore_interaction
from sensai_server.monitoring.recorder import InteractionRecorder

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from uuid import UUID

    from sensai_server.llm.provider import ProviderManager
    from sensai_server.llm.responses import ChatResponse
    from sensai_server.memory.summary import SummaryRepository
    from sensai_server.monitoring.budget import ContextBudget
    from sensai_server.monitoring.interaction import Interaction
    from sensai_server.pipeline.chain import ChatContext, Next

# Given after the conversation: placed before it, a small model reads the conversation as one to go on with, and
# answers its last message instead of summarising it.
SUMMARY_INSTRUCTION = (
    "Summarise the conversation above in a few lines, in the language it is written in, so that it can go on "
    "without its full text. First list every fact the user gave about themselves and their request, exactly as "
    "stated: names, places, figures, dates, durations, decisions. Then, in one line, what was discussed and what "
    "is still open. Write the summary only."
)
SUMMARY_HEADER = "Summary of the earlier conversation, standing for its first messages:"


class CompressionError(SensaiError):
    """Raised when the model could not summarise the conversation."""


@dataclass(frozen=True)
class Compression:
    """Outcome of one compression: how many messages were summarised, and their size before and after."""

    messages: int
    tokens_before: int | None
    """Tokens of the summarised text, as the model read it."""
    tokens_after: int | None
    """Tokens of the summary the model wrote."""

    def describe(self) -> str:
        """Tell the user what was summarised, and how much it saved when the provider reported it."""
        text = f"Summarised the {self.messages} oldest messages of the conversation"
        if self.tokens_before is None or self.tokens_after is None:
            return f"{text}."
        return f"{text}: {self.tokens_before} -> {self.tokens_after} tokens."


class Compressor:
    """Condense the oldest messages of a branch into a summary the model reads in their place.

    The messages stay in the branch: only the summary is stored, with the number of messages it stands for, so it
    is written once and extended by the next compression instead of rewritten from the start.

    One compressor serves every session: the model to summarise with comes from each call. The summaries are read
    and written in a thread, and the model streams through one, so the shared event loop never blocks.
    """

    def __init__(
        self,
        provider_manager: ProviderManager,
        summaries: SummaryRepository,
        keep_recent: int,
        *,
        context_window: int | None = None,
        on_interaction: Callable[[Interaction], None] = ignore_interaction,
    ) -> None:
        """Summarise with the models of `provider_manager`, into `summaries`, all but the `keep_recent` last messages.

        `context_window` is asked of the model like for a chat. Every call is passed to `on_interaction`, tagged with
        the persona but not with the branch: the summariser's prompt is not the conversation's usage. It is called in
        the `finally` of the call, so it must never raise.
        """
        self._provider_manager = provider_manager
        self._summaries = summaries
        self._keep_recent = keep_recent
        self._context_window = context_window
        self._on_interaction = on_interaction

    async def apply(self, branch_id: UUID, messages: list[dict]) -> list[dict]:
        """Return the `messages` of the branch `branch_id` with the ones its summary stands for replaced by it.

        Raises:
            DatabaseError: If the summary could not be read.

        """
        summary = await asyncio.to_thread(self._summaries.get, branch_id)
        if summary is None:
            return messages
        return [self._summary_message(summary.content), *messages[summary.covered :]]

    async def compress(
        self, branch_id: UUID, messages: list[dict], model: str, persona: str | None = None
    ) -> Compression | None:
        """Extend the summary of the branch `branch_id` to all but the last of its `messages`, written by `model`.

        Returns None when there is nothing new to summarise. The call is recorded with `persona`. Once the summary
        is written, its storage is shielded, so a cancellation does not stop it half-way.

        Raises:
            CompressionError: If the model could not summarise them.
            DatabaseError: If the summary could not be read or stored.

        """
        previous = await asyncio.to_thread(self._summaries.get, branch_id)
        covered = 0 if previous is None else previous.covered
        end = len(messages) - self._keep_recent
        # Cut before a user message, so that no answer is sent without the question it answers.
        while end > covered and messages[end]["role"] != Role.USER.value:
            end -= 1
        if end <= covered:
            return None
        content, metrics = await self._summarise(previous, messages[covered:end], model, persona)
        summary = Summary(covered=end, content=content)
        await asyncio.shield(asyncio.to_thread(self._summaries.save, branch_id, summary))
        return Compression(end - covered, metrics.prompt_tokens, metrics.completion_tokens)

    async def _summarise(
        self, previous: Summary | None, messages: list[dict], model: str, persona: str | None
    ) -> tuple[str, ChatMetrics]:
        """Ask `model` for a summary of the `previous` one followed by `messages`.

        The call is recorded as `ReAct._request` records its own: `ERROR` on an error chunk or a `SensaiError`,
        `INTERRUPTED` when cancelled before its `done` chunk. The cancellation propagates.
        """
        provider = self._provider_manager.get_provider(model)
        if provider is None:
            msg = f"No provider serves model {model}"
            raise CompressionError(msg)

        parts = [] if previous is None else [f"{SUMMARY_HEADER}\n{previous.content}"]
        parts.extend(f"{message['role']}: {message['content']}" for message in messages)
        conversation = "\n\n".join(parts)
        request = [
            {"role": Role.USER.value, "content": f"Conversation:\n\n{conversation}\n\n---\n{SUMMARY_INSTRUCTION}"}
        ]
        payload: dict[str, object] = {"model": model, "messages": request, "think": False}
        if self._context_window is not None:
            payload["options"] = {"num_ctx": self._context_window}

        recorder = InteractionRecorder(model=model, request=request)
        content, metrics = "", ChatMetrics()
        status, error = InteractionStatus.OK, None
        done = False
        try:
            async with aclosing(iterate_in_thread(partial(provider.chat, payload, stream=True))) as chunks:
                async for chunk in chunks:
                    recorder.add_chunk(chunk)
                    done = done or chunk.done
                    if chunk.error is not None:
                        status, error = InteractionStatus.ERROR, chunk.error
                        break
                    content += chunk.content or ""
                    metrics = chunk.metrics or metrics
        except GeneratorExit, asyncio.CancelledError:
            if status is not InteractionStatus.ERROR and not done:
                status = InteractionStatus.INTERRUPTED
            raise
        except SensaiError as err:
            status, error = InteractionStatus.ERROR, str(err)
        finally:
            self._on_interaction(replace(recorder.finish(status, error), persona=persona))

        if error is not None:
            raise CompressionError(error)
        if not content.strip():
            msg = "the model wrote an empty summary"
            raise CompressionError(msg)
        return content.strip(), metrics

    @staticmethod
    def _summary_message(content: str) -> dict:
        return {"role": Role.SYSTEM.value, "content": f"{SUMMARY_HEADER}\n{content}"}


class CompressionStage:
    """Send the model the summary of the oldest messages in their place, summarising more once the budget runs low.

    The compressor and the budget are shared by every session, and the branch, the model and the persona come from
    the context, so one instance serves all.
    """

    def __init__(self, compressor: Compressor, budget: ContextBudget) -> None:
        """Compress with `compressor` whenever the usage `budget` measures is near its limit."""
        self._compressor = compressor
        self._budget = budget

    async def __call__(self, context: ChatContext, next_stage: Next) -> AsyncGenerator[ChatResponse]:
        """Shorten the messages of `context`, and stream what the rest of the chain answers.

        A failed compression is reported to the client, and the chat goes on with the messages it could not shorten.
        """
        if context.branch_id is not None:
            await self._shorten(context, context.branch_id)
        async with aclosing(next_stage(context)) as chunks:
            async for chunk in chunks:
                yield chunk

    async def _shorten(self, context: ChatContext, branch_id: UUID) -> None:
        """Compress the branch when near its budget, then put its summary in place of the messages it covers."""
        try:
            if (await self._budget.usage(context.model, branch_id)).near_limit:
                persona = context.persona.name if context.persona is not None else None
                compression = await self._compressor.compress(branch_id, context.messages, context.model, persona)
                if compression is not None:
                    self._budget.compressed(branch_id, compression)
                    await context.out.send_system_message(compression.describe())
            context.messages = await self._compressor.apply(branch_id, context.messages)
        except SensaiError as err:
            await context.out.send_system_message(f"The conversation could not be compressed: {err.message}")
