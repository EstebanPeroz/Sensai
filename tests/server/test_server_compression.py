import asyncio
import threading
from collections.abc import Iterator
from contextlib import aclosing
from typing import cast
from uuid import UUID, uuid4

import pytest
from sensai_protocol import System
from sensai_server.cache import SessionCache
from sensai_server.llm.error import RequestCallError
from sensai_server.llm.persona import Persona
from sensai_server.llm.provider import ProviderManager
from sensai_server.llm.responses import ChatResponse
from sensai_server.memory.persistent_db.error import DatabaseError
from sensai_server.memory.summary import Summary
from sensai_server.monitoring.budget import ContextBudget, ContextUsage
from sensai_server.monitoring.interaction import Interaction, InteractionStatus
from sensai_server.pipeline import ChatContext, answer_chunk
from sensai_server.pipeline.modules import CacheStage, CompressionStage
from sensai_server.pipeline.modules.compression import (
    SUMMARY_HEADER,
    SUMMARY_INSTRUCTION,
    Compression,
    CompressionError,
    Compressor,
)

from tests.server.fakes import FakeSessionCache, RecordingNext, RecordingOutbound

BRANCH = uuid4()
MODEL = "fake-model"


class FakeProvider:
    """Streams `chunks` for every chat, or raises `error`; with `block` set, waits on it after the first chunk."""

    def __init__(
        self,
        chunks: list[ChatResponse] | None = None,
        error: Exception | None = None,
        block: threading.Event | None = None,
    ) -> None:
        self.payloads: list[dict] = []
        self._chunks = chunks if chunks is not None else summary_chunks("Ada wants a 300 EUR budget.")
        self._error = error
        self._block = block
        self.started = threading.Event()

    def chat(self, payload: dict, *, stream: bool) -> Iterator[ChatResponse]:
        assert stream
        self.payloads.append(payload)
        if self._error is not None:
            raise self._error
        return self._stream()

    def _stream(self) -> Iterator[ChatResponse]:
        for index, chunk in enumerate(self._chunks):
            if index == 1 and self._block is not None:
                self.started.set()
                self._block.wait()
            yield chunk


class FakeProviderManager:
    def __init__(self, provider: FakeProvider) -> None:
        self._provider = provider
        self.asked: list[str] = []

    def get_provider(self, model: str) -> FakeProvider | None:
        self.asked.append(model)
        return self._provider if model == MODEL else None


class FakeSummaries:
    def __init__(self, error: DatabaseError | None = None) -> None:
        self.stored: dict[UUID, Summary] = {}
        self._error = error

    def get(self, branch_id: UUID) -> Summary | None:
        if self._error is not None:
            raise self._error
        return self.stored.get(branch_id)

    def save(self, branch_id: UUID, summary: Summary) -> None:
        self.stored[branch_id] = summary


class FakeBudget:
    """A `ContextBudget` whose usage is near its limit or not, and that records what it was asked."""

    def __init__(self, *, near_limit: bool) -> None:
        self._used = 90 if near_limit else 10
        self.asked: list[tuple[str, UUID | None]] = []

    async def usage(self, model: str, branch_id: UUID | None) -> ContextUsage:
        self.asked.append((model, branch_id))
        return ContextUsage(self._used, 100, 0.8)


def summary_chunks(content: str) -> list[ChatResponse]:
    return [
        ChatResponse({"done": False, "message": {"role": "assistant", "content": content}}),
        ChatResponse(
            {"done": True, "message": {"role": "assistant", "content": ""}, "prompt_eval_count": 900, "eval_count": 40}
        ),
    ]


def conversation(length: int) -> list[dict]:
    return [{"role": "user" if n % 2 == 0 else "assistant", "content": f"message {n}"} for n in range(length)]


def compressor(
    provider: FakeProvider | None = None,
    summaries: FakeSummaries | None = None,
    keep_recent: int = 2,
    interactions: list[Interaction] | None = None,
    context_window: int | None = None,
) -> Compressor:
    return Compressor(
        cast("ProviderManager", FakeProviderManager(provider or FakeProvider())),
        summaries or FakeSummaries(),
        keep_recent,
        context_window=context_window,
        on_interaction=(interactions if interactions is not None else []).append,
    )


class TestCompress:
    async def test_summarises_all_but_the_recent_messages(self) -> None:
        provider, summaries = FakeProvider(), FakeSummaries()

        compression = await compressor(provider, summaries).compress(BRANCH, conversation(6), MODEL)

        assert compression == Compression(messages=4, tokens_before=900, tokens_after=40)
        assert summaries.stored[BRANCH] == Summary(covered=4, content="Ada wants a 300 EUR budget.")
        assert provider.payloads[0]["model"] == MODEL
        assert provider.payloads[0]["think"] is False
        assert provider.payloads[0]["messages"] == [
            {
                "role": "user",
                "content": "Conversation:\n\nuser: message 0\n\nassistant: message 1\n\nuser: message 2\n\n"
                f"assistant: message 3\n\n---\n{SUMMARY_INSTRUCTION}",
            }
        ]

    async def test_extends_the_previous_summary_with_the_messages_after_it(self) -> None:
        provider, summaries = FakeProvider(), FakeSummaries()
        summaries.stored[BRANCH] = Summary(covered=4, content="Ada introduced herself.")

        compression = await compressor(provider, summaries).compress(BRANCH, conversation(8), MODEL)

        assert compression is not None
        assert compression.messages == 2
        assert summaries.stored[BRANCH].covered == 6
        assert provider.payloads[0]["messages"][0]["content"].startswith(
            f"Conversation:\n\n{SUMMARY_HEADER}\nAda introduced herself.\n\nuser: message 4\n\nassistant: message 5\n\n"
        )

    @pytest.mark.parametrize(("keep_recent", "covered"), [(3, 2), (4, 2), (5, 0)])
    async def test_cuts_before_a_user_message(self, keep_recent: int, covered: int) -> None:
        summaries = FakeSummaries()

        compression = await compressor(summaries=summaries, keep_recent=keep_recent).compress(
            BRANCH, conversation(6), MODEL
        )

        if covered == 0:
            assert compression is None
        else:
            assert summaries.stored[BRANCH].covered == covered

    @pytest.mark.parametrize("length", [2, 6])
    async def test_does_nothing_without_new_messages_to_summarise(self, length: int) -> None:
        provider, summaries = FakeProvider(), FakeSummaries()
        summaries.stored[BRANCH] = Summary(covered=4, content="Ada introduced herself.")

        assert await compressor(provider, summaries).compress(BRANCH, conversation(length), MODEL) is None
        assert provider.payloads == []

    async def test_asks_for_the_context_window(self) -> None:
        provider = FakeProvider()

        await compressor(provider, context_window=8192).compress(BRANCH, conversation(6), MODEL)

        assert provider.payloads[0]["options"] == {"num_ctx": 8192}

    async def test_leaves_the_context_window_to_the_provider_by_default(self) -> None:
        provider = FakeProvider()

        await compressor(provider).compress(BRANCH, conversation(6), MODEL)

        assert "options" not in provider.payloads[0]

    async def test_a_model_no_provider_serves_raises(self) -> None:
        with pytest.raises(CompressionError, match="No provider serves model other"):
            await compressor().compress(BRANCH, conversation(6), "other")

    async def test_a_model_error_keeps_the_previous_summary(self) -> None:
        error_chunk = ChatResponse({"error": "model not found"})
        summaries, interactions = FakeSummaries(), []
        summaries.stored[BRANCH] = Summary(covered=2, content="Ada introduced herself.")

        with pytest.raises(CompressionError, match="model not found"):
            await compressor(FakeProvider([error_chunk]), summaries, interactions=interactions).compress(
                BRANCH, conversation(6), MODEL
            )

        assert summaries.stored[BRANCH] == Summary(covered=2, content="Ada introduced herself.")
        assert [interaction.status for interaction in interactions] == [InteractionStatus.ERROR]

    async def test_a_connection_failure_raises(self) -> None:
        with pytest.raises(CompressionError, match="refused"):
            await compressor(FakeProvider(error=RequestCallError("refused"))).compress(BRANCH, conversation(6), MODEL)

    async def test_an_empty_summary_is_not_stored(self) -> None:
        summaries = FakeSummaries()

        with pytest.raises(CompressionError, match="empty summary"):
            await compressor(FakeProvider(summary_chunks("  ")), summaries).compress(BRANCH, conversation(6), MODEL)

        assert summaries.stored == {}


class TestRecording:
    async def test_records_one_ok_call_with_the_persona_and_no_branch(self) -> None:
        interactions: list[Interaction] = []

        await compressor(interactions=interactions).compress(BRANCH, conversation(6), MODEL, "tutor")

        assert len(interactions) == 1
        interaction = interactions[0]
        assert interaction.status is InteractionStatus.OK
        assert interaction.response == "Ada wants a 300 EUR budget."
        assert interaction.model == MODEL
        assert interaction.persona == "tutor"
        assert interaction.branch_id is None

    async def test_a_provider_error_is_recorded_as_an_error(self) -> None:
        interactions: list[Interaction] = []

        with pytest.raises(CompressionError):
            await compressor(FakeProvider(error=RequestCallError("refused")), interactions=interactions).compress(
                BRANCH, conversation(6), MODEL
            )

        assert [interaction.status for interaction in interactions] == [InteractionStatus.ERROR]
        assert interactions[0].error is not None
        assert "refused" in interactions[0].error

    async def test_a_cancel_mid_summary_is_interrupted_stores_nothing_and_propagates(self) -> None:
        block = threading.Event()
        provider, summaries, interactions = FakeProvider(block=block), FakeSummaries(), []
        task = asyncio.create_task(
            compressor(provider, summaries, interactions=interactions).compress(BRANCH, conversation(6), MODEL)
        )
        try:
            await asyncio.to_thread(provider.started.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            block.set()

        assert [interaction.status for interaction in interactions] == [InteractionStatus.INTERRUPTED]
        assert interactions[0].branch_id is None
        assert summaries.stored == {}


class TestApply:
    async def test_replaces_the_messages_the_summary_stands_for(self) -> None:
        summaries = FakeSummaries()
        summaries.stored[BRANCH] = Summary(covered=4, content="Ada introduced herself.")

        messages = await compressor(summaries=summaries).apply(BRANCH, conversation(6))

        assert messages == [
            {"role": "system", "content": f"{SUMMARY_HEADER}\nAda introduced herself."},
            *conversation(6)[4:],
        ]

    async def test_leaves_the_messages_of_a_branch_without_summary(self) -> None:
        assert await compressor().apply(BRANCH, conversation(6)) == conversation(6)


class TestCompressionDescription:
    def test_tells_the_tokens_saved(self) -> None:
        assert Compression(4, 900, 40).describe() == (
            "Summarised the 4 oldest messages of the conversation: 900 -> 40 tokens."
        )

    def test_without_token_counts(self) -> None:
        assert Compression(4, None, None).describe() == "Summarised the 4 oldest messages of the conversation."


def _context(messages: list[dict], branch_id: UUID | None = BRANCH, persona: Persona | None = None) -> ChatContext:
    return ChatContext(
        message="last", out=RecordingOutbound(), model=MODEL, messages=messages, branch_id=branch_id, persona=persona
    )


async def run_stage(stage: CompressionStage, context: ChatContext) -> tuple[list[str], list[dict], list[ChatResponse]]:
    next_stage = RecordingNext(answer_chunk("answer"))
    async with aclosing(stage(context, next_stage)) as chunks:
        received = [chunk async for chunk in chunks]
    out = cast("RecordingOutbound", context.out)
    system = [message.message for message in out.sent if isinstance(message, System)]
    return system, next_stage.seen_messages[0], received


def stage(compressor: Compressor, *, near_limit: bool) -> CompressionStage:
    return CompressionStage(compressor, cast("ContextBudget", FakeBudget(near_limit=near_limit)))


class TestCompressionStage:
    async def test_compresses_near_the_limit_and_sends_the_summary(self) -> None:
        system, seen, received = await run_stage(stage(compressor(), near_limit=True), _context(conversation(6)))

        assert seen == [
            {"role": "system", "content": f"{SUMMARY_HEADER}\nAda wants a 300 EUR budget."},
            *conversation(6)[4:],
        ]
        assert system == ["Summarised the 4 oldest messages of the conversation: 900 -> 40 tokens."]
        assert [chunk.content for chunk in received] == ["answer"]

    async def test_asks_the_budget_for_the_context_model_and_branch(self) -> None:
        budget = FakeBudget(near_limit=False)

        await run_stage(CompressionStage(compressor(), cast("ContextBudget", budget)), _context(conversation(6)))

        assert budget.asked == [(MODEL, BRANCH)]

    async def test_records_the_summary_with_the_context_persona(self) -> None:
        interactions: list[Interaction] = []
        persona = Persona(uuid=None, name="tutor", description="A tutor.", prompt="Teach.")

        await run_stage(
            stage(compressor(interactions=interactions), near_limit=True), _context(conversation(6), persona=persona)
        )

        assert [(each.persona, each.branch_id) for each in interactions] == [("tutor", None)]

    async def test_sends_the_stored_summary_without_compressing_below_the_limit(self) -> None:
        provider, summaries = FakeProvider(), FakeSummaries()
        summaries.stored[BRANCH] = Summary(covered=2, content="Ada introduced herself.")

        system, seen, _ = await run_stage(
            stage(compressor(provider, summaries), near_limit=False), _context(conversation(6))
        )

        assert seen[0] == {"role": "system", "content": f"{SUMMARY_HEADER}\nAda introduced herself."}
        assert len(seen) == 5
        assert provider.payloads == []
        assert system == []

    async def test_leaves_the_messages_below_the_limit_without_summary(self) -> None:
        provider = FakeProvider()

        system, seen, _ = await run_stage(stage(compressor(provider), near_limit=False), _context(conversation(6)))

        assert seen == conversation(6)
        assert provider.payloads == []
        assert system == []

    async def test_sends_nothing_when_there_is_nothing_new_to_summarise(self) -> None:
        provider = FakeProvider()

        system, seen, _ = await run_stage(
            stage(compressor(provider, keep_recent=5), near_limit=True), _context(conversation(6))
        )

        assert seen == conversation(6)
        assert system == []

    async def test_leaves_a_conversation_without_branch(self) -> None:
        provider = FakeProvider()
        budget = FakeBudget(near_limit=True)

        _, seen, _ = await run_stage(
            CompressionStage(compressor(provider), cast("ContextBudget", budget)),
            _context(conversation(6), branch_id=None),
        )

        assert seen == conversation(6)
        assert provider.payloads == []
        assert budget.asked == []

    async def test_a_failed_compression_goes_on_with_the_full_messages(self) -> None:
        interactions: list[Interaction] = []
        failing = compressor(FakeProvider(error=RequestCallError("refused")), interactions=interactions)

        system, seen, received = await run_stage(stage(failing, near_limit=True), _context(conversation(6)))

        assert seen == conversation(6)
        assert len(system) == 1
        assert system[0].startswith("The conversation could not be compressed:")
        assert system[0].endswith("refused")
        assert [interaction.status for interaction in interactions] == [InteractionStatus.ERROR]
        assert [chunk.content for chunk in received] == ["answer"]

    async def test_an_unreadable_summary_goes_on_with_the_full_messages(self) -> None:
        unreadable = compressor(summaries=FakeSummaries(DatabaseError("down")))

        system, seen, _ = await run_stage(stage(unreadable, near_limit=False), _context(conversation(6)))

        assert seen == conversation(6)
        assert len(system) == 1
        assert system[0].startswith("The conversation could not be compressed:")
        assert system[0].endswith("down")


class TestAfterTheCache:
    async def test_a_cache_hit_never_compresses(self) -> None:
        provider, budget = FakeProvider(), FakeBudget(near_limit=True)
        compression = CompressionStage(compressor(provider), cast("ContextBudget", budget))
        context = _context(conversation(6))
        context.cache = cast("SessionCache", FakeSessionCache({"last": "cached"}))
        next_stage = RecordingNext(answer_chunk("model"))

        async def chain(context: ChatContext):  # noqa: ANN202 - an async generator relaying the compression stage
            async with aclosing(compression(context, next_stage)) as chunks:
                async for chunk in chunks:
                    yield chunk

        async with aclosing(CacheStage()(context, chain)) as chunks:
            received = [chunk async for chunk in chunks]

        assert [chunk.content for chunk in received] == ["cached"]
        assert provider.payloads == []
        assert budget.asked == []
        assert context.messages == conversation(6)
