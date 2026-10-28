from collections.abc import Iterator
from uuid import UUID, uuid4

import pytest

from sensai.llm.error import RequestCallError
from sensai.llm.responses import ChatResponse
from sensai.memory.persistent_db.error import DatabaseError
from sensai.memory.summary import Summary
from sensai.monitoring.interaction import Interaction, InteractionStatus
from sensai.pipeline.modules.compression import (
    SUMMARY_HEADER,
    SUMMARY_INSTRUCTION,
    Compression,
    CompressionError,
    Compressor,
)

BRANCH = uuid4()


class FakeProvider:
    def __init__(self, chunks: list[ChatResponse] | None = None, error: Exception | None = None) -> None:
        self.payloads: list[dict] = []
        self._chunks = chunks if chunks is not None else summary_chunks("Ada wants a 300 EUR budget.")
        self._error = error

    def current_model(self) -> str:
        return "fake-model"

    def chat(self, payload: dict, *, stream: bool) -> Iterator[ChatResponse]:
        assert stream
        self.payloads.append(payload)
        if self._error is not None:
            raise self._error
        return iter(self._chunks)


class FakeProviderManager:
    def __init__(self, provider: FakeProvider) -> None:
        self._provider = provider

    def provider(self) -> FakeProvider:
        return self._provider


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
        FakeProviderManager(provider or FakeProvider()),  # type: ignore[arg-type]
        summaries or FakeSummaries(),
        keep_recent,
        context_window=context_window,
        on_interaction=(interactions if interactions is not None else []).append,
    )


class TestCompress:
    def test_summarises_all_but_the_recent_messages(self) -> None:
        provider, summaries = FakeProvider(), FakeSummaries()

        compression = compressor(provider, summaries).compress(BRANCH, conversation(6))

        assert compression == Compression(messages=4, tokens_before=900, tokens_after=40)
        assert summaries.stored[BRANCH] == Summary(covered=4, content="Ada wants a 300 EUR budget.")
        assert provider.payloads[0]["messages"] == [
            {
                "role": "user",
                "content": "Conversation:\n\nuser: message 0\n\nassistant: message 1\n\nuser: message 2\n\n"
                f"assistant: message 3\n\n---\n{SUMMARY_INSTRUCTION}",
            }
        ]

    def test_extends_the_previous_summary_with_the_messages_after_it(self) -> None:
        provider, summaries = FakeProvider(), FakeSummaries()
        summaries.stored[BRANCH] = Summary(covered=4, content="Ada introduced herself.")

        compression = compressor(provider, summaries).compress(BRANCH, conversation(8))

        assert compression is not None
        assert compression.messages == 2
        assert summaries.stored[BRANCH].covered == 6
        assert provider.payloads[0]["messages"][0]["content"].startswith(
            f"Conversation:\n\n{SUMMARY_HEADER}\nAda introduced herself.\n\nuser: message 4\n\nassistant: message 5\n\n"
        )

    @pytest.mark.parametrize(("keep_recent", "covered"), [(3, 2), (4, 2), (5, 0)])
    def test_cuts_before_a_user_message(self, keep_recent: int, covered: int) -> None:
        summaries = FakeSummaries()

        compression = compressor(summaries=summaries, keep_recent=keep_recent).compress(BRANCH, conversation(6))

        if covered == 0:
            assert compression is None
        else:
            assert summaries.stored[BRANCH].covered == covered

    @pytest.mark.parametrize("length", [2, 6])
    def test_does_nothing_without_new_messages_to_summarise(self, length: int) -> None:
        provider, summaries = FakeProvider(), FakeSummaries()
        summaries.stored[BRANCH] = Summary(covered=4, content="Ada introduced herself.")

        assert compressor(provider, summaries).compress(BRANCH, conversation(length)) is None
        assert provider.payloads == []

    def test_asks_for_the_context_window(self) -> None:
        provider = FakeProvider()

        compressor(provider, context_window=8192).compress(BRANCH, conversation(6))

        assert provider.payloads[0]["options"] == {"num_ctx": 8192}

    def test_records_the_call(self) -> None:
        interactions: list[Interaction] = []

        compressor(interactions=interactions).compress(BRANCH, conversation(6))

        assert [interaction.status for interaction in interactions] == [InteractionStatus.OK]
        assert interactions[0].response == "Ada wants a 300 EUR budget."

    def test_a_model_error_keeps_the_previous_summary(self) -> None:
        error_chunk = ChatResponse({"error": "model not found"})
        summaries, interactions = FakeSummaries(), []
        summaries.stored[BRANCH] = Summary(covered=2, content="Ada introduced herself.")

        with pytest.raises(CompressionError, match="model not found"):
            compressor(FakeProvider([error_chunk]), summaries, interactions=interactions).compress(
                BRANCH, conversation(6)
            )

        assert summaries.stored[BRANCH] == Summary(covered=2, content="Ada introduced herself.")
        assert [interaction.status for interaction in interactions] == [InteractionStatus.ERROR]

    def test_a_connection_failure_raises(self) -> None:
        with pytest.raises(CompressionError, match="refused"):
            compressor(FakeProvider(error=RequestCallError("refused"))).compress(BRANCH, conversation(6))

    def test_an_empty_summary_is_not_stored(self) -> None:
        summaries = FakeSummaries()

        with pytest.raises(CompressionError, match="empty summary"):
            compressor(FakeProvider(summary_chunks("  ")), summaries).compress(BRANCH, conversation(6))

        assert summaries.stored == {}


class TestCompressionDescription:
    def test_tells_the_tokens_saved(self) -> None:
        assert Compression(4, 900, 40).describe() == (
            "Summarised the 4 oldest messages of the conversation: 900 -> 40 tokens."
        )

    def test_without_token_counts(self) -> None:
        assert Compression(4, None, None).describe() == "Summarised the 4 oldest messages of the conversation."
