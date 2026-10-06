from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from sensai.history.conversation import CACHE_CONTEXT_TURNS, DEFAULT_PERSONA, Conversation
from sensai.llm.responses import ChatResponse
from sensai.memory.semantic_cache_repository import CacheScope

if TYPE_CHECKING:
    from collections.abc import Iterator

MODEL = "qwen3:1.7b"
SCOPE = CacheScope(model=MODEL, persona=DEFAULT_PERSONA)


class FakeUI:
    def __init__(self) -> None:
        self.responses: list[ChatResponse] = []

    def send_user_input(self, _response: str) -> bool:
        return True

    def send_ai_response(self, response: ChatResponse) -> bool:
        self.responses.append(response)
        return True

    def wait_event(self, *, timeout: float | None = None) -> bool:  # noqa: ARG002
        return False

    def text(self) -> str:
        return "".join(response.content or "" for response in self.responses)


class FakeCache:
    def __init__(self, answers: dict[tuple[str, str], str] | None = None) -> None:
        self.answers = answers or {}
        self.lookups: list[tuple[str, CacheScope, str]] = []
        self.stored: list[tuple[str, CacheScope, str, str]] = []

    def lookup(self, query: str, scope: CacheScope, context: str = "") -> str | None:
        self.lookups.append((query, scope, context))
        return self.answers.get((query, context))

    def store(self, query: str, scope: CacheScope, answer: str, context: str = "") -> None:
        self.stored.append((query, scope, answer, context))

    def clear(self) -> None:
        self.answers.clear()


def chunks(*contents: str, error: str | None = None) -> Iterator[ChatResponse]:
    for content in contents:
        yield ChatResponse({"done": False, "message": {"role": "assistant", "content": content}})
    if error is not None:
        yield ChatResponse({"error": error})
    else:
        yield ChatResponse({"done": True, "message": {"role": "assistant", "content": ""}})


@pytest.fixture
def provider() -> MagicMock:
    provider = MagicMock()
    provider.chat.side_effect = lambda *_args, **_kwargs: chunks("Redis is ", "a store.")
    return provider


class TestWithoutCache:
    def test_answers_from_the_provider(self, provider: MagicMock) -> None:
        ui = FakeUI()

        Conversation(provider, MODEL).chat("What is Redis?", ui)  # type: ignore[arg-type]

        assert ui.text() == "Redis is a store."


class TestFirstTurn:
    def test_miss_asks_the_provider_and_stores_the_answer(self, provider: MagicMock) -> None:
        cache, ui = FakeCache(), FakeUI()

        Conversation(provider, MODEL, cache).chat("What is Redis?", ui)  # type: ignore[arg-type]

        provider.chat.assert_called_once()
        assert cache.lookups == [("What is Redis?", SCOPE, "")]
        assert cache.stored == [("What is Redis?", SCOPE, "Redis is a store.", "")]
        assert ui.text() == "Redis is a store."

    def test_hit_answers_without_the_provider(self, provider: MagicMock) -> None:
        cache, ui = FakeCache({("What is Redis?", ""): "From the cache."}), FakeUI()

        Conversation(provider, MODEL, cache).chat("What is Redis?", ui)  # type: ignore[arg-type]

        provider.chat.assert_not_called()
        assert ui.text() == "From the cache."
        assert ui.responses[-1].done
        assert cache.stored == []

    def test_hit_is_kept_in_the_history(self, provider: MagicMock) -> None:
        cache, ui = FakeCache({("What is Redis?", ""): "From the cache."}), FakeUI()
        conversation = Conversation(provider, MODEL, cache)  # type: ignore[arg-type]

        conversation.chat("What is Redis?", ui)  # type: ignore[arg-type]
        conversation.chat("And Memcached?", ui)  # type: ignore[arg-type]

        messages = provider.chat.call_args.args[0]["messages"]
        assert [message["content"] for message in messages] == ["What is Redis?", "From the cache.", "And Memcached?"]

    def test_failed_answer_is_not_stored(self, provider: MagicMock) -> None:
        provider.chat.side_effect = lambda *_args, **_kwargs: chunks("Redis", error="model crashed")
        cache = FakeCache()

        Conversation(provider, MODEL, cache).chat("What is Redis?", FakeUI())  # type: ignore[arg-type]

        assert cache.stored == []

    def test_empty_answer_is_not_stored(self, provider: MagicMock) -> None:
        provider.chat.side_effect = lambda *_args, **_kwargs: chunks(" ")
        cache = FakeCache()

        Conversation(provider, MODEL, cache).chat("What is Redis?", FakeUI())  # type: ignore[arg-type]

        assert cache.stored == []


class TestFollowUp:
    def test_miss_is_stored_with_the_previous_user_message(self, provider: MagicMock) -> None:
        cache = FakeCache()
        conversation = Conversation(provider, MODEL, cache)  # type: ignore[arg-type]
        conversation.chat("What is Redis?", FakeUI())  # type: ignore[arg-type]

        conversation.chat("Give me an example.", FakeUI())  # type: ignore[arg-type]

        assert cache.lookups[-1] == ("Give me an example.", SCOPE, "What is Redis?")
        assert cache.stored[-1] == ("Give me an example.", SCOPE, "Redis is a store.", "What is Redis?")

    def test_hit_answers_without_the_provider(self, provider: MagicMock) -> None:
        cache = FakeCache({("Give me an example.", "What is Redis?"): "SET key value"})
        conversation = Conversation(provider, MODEL, cache)  # type: ignore[arg-type]
        conversation.chat("What is Redis?", FakeUI())  # type: ignore[arg-type]
        ui = FakeUI()

        conversation.chat("Give me an example.", ui)  # type: ignore[arg-type]

        provider.chat.assert_called_once()
        assert ui.text() == "SET key value"

    def test_same_query_in_another_context_misses(self, provider: MagicMock) -> None:
        cache = FakeCache({("Give me an example.", "What is Redis?"): "SET key value"})
        conversation = Conversation(provider, MODEL, cache)  # type: ignore[arg-type]
        conversation.chat("What is PostgreSQL?", FakeUI())  # type: ignore[arg-type]
        ui = FakeUI()

        conversation.chat("Give me an example.", ui)  # type: ignore[arg-type]

        assert ui.text() == "Redis is a store."

    def test_context_keeps_the_last_user_messages_only(self, provider: MagicMock) -> None:
        cache = FakeCache()
        conversation = Conversation(provider, MODEL, cache)  # type: ignore[arg-type]
        for query in ("first", "second", "third"):
            conversation.chat(query, FakeUI())  # type: ignore[arg-type]

        conversation.chat("fourth", FakeUI())  # type: ignore[arg-type]

        assert cache.lookups[-1] == ("fourth", SCOPE, "\n".join(["second", "third"][-CACHE_CONTEXT_TURNS:]))
