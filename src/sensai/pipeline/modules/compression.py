from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sensai.error import SensaiError
from sensai.llm.message import Role
from sensai.llm.responses import ChatMetrics
from sensai.memory.summary import Summary
from sensai.monitoring.interaction import InteractionStatus, ignore_interaction
from sensai.monitoring.recorder import InteractionRecorder

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from uuid import UUID

    from sensai.llm.provider_manager import ProviderManager
    from sensai.llm.responses import ChatResponse
    from sensai.memory.summary import SummaryRepository
    from sensai.monitoring.interaction import Interaction
    from sensai.pipeline.chain import ChatContext, Next

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
        """Summarise with the current model, into `summaries`, all but the `keep_recent` last messages.

        `context_window` is asked of the model like for a chat; every call is passed to `on_interaction`.
        """
        self._provider_manager = provider_manager
        self._summaries = summaries
        self._keep_recent = keep_recent
        self._context_window = context_window
        self._on_interaction = on_interaction

    def apply(self, branch_id: UUID, messages: list[dict]) -> list[dict]:
        """Return the `messages` of the branch `branch_id` with the ones its summary stands for replaced by it.

        Raises:
            DatabaseError: If the summary could not be read.

        """
        summary = self._summaries.get(branch_id)
        if summary is None:
            return messages
        return [self._summary_message(summary.content), *messages[summary.covered :]]

    def compress(self, branch_id: UUID, messages: list[dict]) -> Compression | None:
        """Extend the summary of the branch `branch_id` to all but the last of its `messages`.

        Returns None when there is nothing new to summarise.

        Raises:
            CompressionError: If the model could not summarise them.
            DatabaseError: If the summary could not be read or stored.

        """
        previous = self._summaries.get(branch_id)
        covered = 0 if previous is None else previous.covered
        end = len(messages) - self._keep_recent
        # Cut before a user message, so that no answer is sent without the question it answers.
        while end > covered and messages[end]["role"] != Role.USER.value:
            end -= 1
        if end <= covered:
            return None
        content, metrics = self._summarise(previous, messages[covered:end])
        self._summaries.save(branch_id, Summary(covered=end, content=content))
        return Compression(end - covered, metrics.prompt_tokens, metrics.completion_tokens)

    def _summarise(self, previous: Summary | None, messages: list[dict]) -> tuple[str, ChatMetrics]:
        """Ask the model for a summary of the `previous` one followed by `messages`."""
        parts = [] if previous is None else [f"{SUMMARY_HEADER}\n{previous.content}"]
        parts.extend(f"{message['role']}: {message['content']}" for message in messages)
        conversation = "\n\n".join(parts)
        request = [
            {"role": Role.USER.value, "content": f"Conversation:\n\n{conversation}\n\n---\n{SUMMARY_INSTRUCTION}"}
        ]
        provider = self._provider_manager.provider()
        model = provider.current_model()
        payload: dict = {"model": model, "messages": request, "think": False}
        if self._context_window is not None:
            payload["options"] = {"num_ctx": self._context_window}

        recorder = InteractionRecorder(model=model, request=request)
        content, metrics = "", ChatMetrics()
        status, error = InteractionStatus.OK, None
        try:
            for chunk in provider.chat(payload, stream=True):
                recorder.add_chunk(chunk)
                if chunk.error is not None:
                    status, error = InteractionStatus.ERROR, chunk.error
                    break
                content += chunk.content or ""
                metrics = chunk.metrics or metrics
        except SensaiError as err:
            status, error = InteractionStatus.ERROR, str(err)
        finally:
            self._on_interaction(recorder.finish(status, error))

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
    """Send the model the summary of the oldest messages in their place, summarising more once the budget runs low."""

    def __init__(self, compressor: Compressor, should_compress: Callable[[], bool]) -> None:
        """Compress with `compressor` whenever `should_compress` says the context is near its budget."""
        self._compressor = compressor
        self._should_compress = should_compress

    def __call__(self, context: ChatContext, next_stage: Next) -> Iterator[ChatResponse]:
        """Shorten the messages of `context`, and stream what the rest of the chain answers.

        A failed compression is reported to the UI, and the chat goes on with the messages it could not shorten.
        """
        if context.branch_id is None:
            return next_stage(context)
        try:
            if self._should_compress():
                compression = self._compressor.compress(context.branch_id, context.messages)
                if compression is not None:
                    context.ui.send_system_message(compression.describe())
            context.messages = self._compressor.apply(context.branch_id, context.messages)
        except SensaiError as err:
            context.ui.send_system_message(f"The conversation could not be compressed: {err.message}")
        return next_stage(context)
