from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from sensai.core.events import EventBus
from sensai.history.conversation import Conversation
from sensai.llm.message import Role
from sensai.llm.persona import Persona
from sensai.llm.responses import ChatResponse
from sensai.llm.thinking.react import ReAct
from sensai.pipeline import ChatPipeline
from sensai.pipeline.modules import ModelStage
from sensai.semantic.cache_key import CACHE_CONTEXT_TURNS, NO_PERSONA
from sensai.semantic.cache_repository import CacheScope

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

MODEL = "qwen3:1.7b"
SCOPE = CacheScope(model=MODEL, persona=NO_PERSONA)


class FakeUI:
    def __init__(self) -> None:
        self.responses: list[ChatResponse] = []

    def send_input(self, _role: Role, _response: str) -> bool:
        return True

    def send_error(self, _message: str) -> bool:
        return True

    def send_system_message(self, _message: str) -> bool:
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


def chunks(*contents: str, error: str | None = None) -> Iterator[ChatResponse]:
    for content in contents:
        yield ChatResponse({"done": False, "message": {"role": "assistant", "content": content}})
    if error is not None:
        yield ChatResponse({"error": error})
    else:
        yield ChatResponse({"done": True, "message": {"role": "assistant", "content": ""}})


@pytest.fixture(autouse=True)
def db() -> Iterator[MagicMock]:
    db = MagicMock()
    db.add_message_to_branch.side_effect = lambda *_args: uuid4()
    with patch("sensai.history.conversation.get_db", return_value=db):
        yield db


@pytest.fixture
def react() -> MagicMock:
    react = MagicMock()
    react.generate_answer.side_effect = lambda _messages, _on_interaction: chunks("Redis is ", "a store.")
    return react


@pytest.fixture
def provider_manager() -> MagicMock:
    provider_manager = MagicMock()
    provider_manager.provider.return_value.current_model.return_value = MODEL
    return provider_manager


@pytest.fixture
def persona_manager() -> MagicMock:
    persona_manager = MagicMock()
    persona_manager.active_persona.return_value = None
    return persona_manager


@pytest.fixture
def make_conversation(
    react: MagicMock, provider_manager: MagicMock, persona_manager: MagicMock
) -> Callable[[FakeCache | None], Conversation]:
    def make(cache: FakeCache | None = None) -> Conversation:
        pipeline = ChatPipeline(ModelStage(react))
        conversation = Conversation(provider_manager, persona_manager, pipeline, EventBus(), cache)
        conversation.uuid = uuid4()
        return conversation

    return make


class TestWithoutCache:
    def test_answers_from_the_provider(self, make_conversation: Callable[..., Conversation]) -> None:
        ui = FakeUI()

        make_conversation().chat("What is Redis?", ui)

        assert ui.text() == "Redis is a store."


class TestFirstTurn:
    def test_miss_asks_the_provider_and_stores_the_answer(
        self, make_conversation: Callable[..., Conversation], react: MagicMock
    ) -> None:
        cache, ui = FakeCache(), FakeUI()

        make_conversation(cache).chat("What is Redis?", ui)

        react.generate_answer.assert_called_once()
        assert cache.lookups == [("What is Redis?", SCOPE, "")]
        assert cache.stored == [("What is Redis?", SCOPE, "Redis is a store.", "")]
        assert ui.text() == "Redis is a store."

    def test_hit_answers_without_the_provider(
        self, make_conversation: Callable[..., Conversation], react: MagicMock
    ) -> None:
        cache, ui = FakeCache({("What is Redis?", ""): "From the cache."}), FakeUI()

        make_conversation(cache).chat("What is Redis?", ui)

        react.generate_answer.assert_not_called()
        assert ui.text() == "From the cache."
        assert ui.responses[-1].done
        assert cache.stored == []

    def test_hit_is_kept_in_the_history(self, make_conversation: Callable[..., Conversation], react: MagicMock) -> None:
        cache, ui = FakeCache({("What is Redis?", ""): "From the cache."}), FakeUI()
        conversation = make_conversation(cache)

        conversation.chat("What is Redis?", ui)
        conversation.chat("And Memcached?", ui)

        messages = react.generate_answer.call_args.args[0]
        assert [message["content"] for message in messages] == ["What is Redis?", "From the cache.", "And Memcached?"]

    def test_failed_answer_is_not_stored(
        self, make_conversation: Callable[..., Conversation], react: MagicMock
    ) -> None:
        react.generate_answer.side_effect = lambda _messages, _on_interaction: chunks("Redis", error="model crashed")
        cache = FakeCache()

        make_conversation(cache).chat("What is Redis?", FakeUI())

        assert cache.stored == []

    def test_answer_built_with_tools_is_not_stored(
        self, make_conversation: Callable[..., Conversation], react: MagicMock
    ) -> None:
        tool_call = {"function": {"name": "web_search", "arguments": {"query": "redis"}}}
        react.generate_answer.side_effect = lambda _messages, _on_interaction: iter(
            [
                ChatResponse({"done": False, "message": {"role": "assistant", "tool_calls": [tool_call]}}),
                *chunks("Redis"),
            ]
        )
        cache = FakeCache()

        make_conversation(cache).chat("What is Redis?", FakeUI())

        assert cache.stored == []

    def test_empty_answer_is_not_stored(self, make_conversation: Callable[..., Conversation], react: MagicMock) -> None:
        react.generate_answer.side_effect = lambda _messages, _on_interaction: chunks(" ")
        cache = FakeCache()

        make_conversation(cache).chat("What is Redis?", FakeUI())

        assert cache.stored == []


class TestFollowUp:
    def test_miss_is_stored_with_the_previous_user_message(
        self, make_conversation: Callable[..., Conversation]
    ) -> None:
        cache = FakeCache()
        conversation = make_conversation(cache)
        conversation.chat("What is Redis?", FakeUI())

        conversation.chat("Give me an example.", FakeUI())

        assert cache.lookups[-1] == ("Give me an example.", SCOPE, "What is Redis?")
        assert cache.stored[-1] == ("Give me an example.", SCOPE, "Redis is a store.", "What is Redis?")

    def test_hit_answers_without_the_provider(
        self, make_conversation: Callable[..., Conversation], react: MagicMock
    ) -> None:
        cache = FakeCache({("Give me an example.", "What is Redis?"): "SET key value"})
        conversation = make_conversation(cache)
        conversation.chat("What is Redis?", FakeUI())
        ui = FakeUI()

        conversation.chat("Give me an example.", ui)

        react.generate_answer.assert_called_once()
        assert ui.text() == "SET key value"

    def test_same_query_in_another_context_misses(self, make_conversation: Callable[..., Conversation]) -> None:
        cache = FakeCache({("Give me an example.", "What is Redis?"): "SET key value"})
        conversation = make_conversation(cache)
        conversation.chat("What is PostgreSQL?", FakeUI())
        ui = FakeUI()

        conversation.chat("Give me an example.", ui)

        assert ui.text() == "Redis is a store."

    def test_context_keeps_the_last_user_messages_only(self, make_conversation: Callable[..., Conversation]) -> None:
        cache = FakeCache()
        conversation = make_conversation(cache)
        for query in ("first", "second", "third"):
            conversation.chat(query, FakeUI())

        conversation.chat("fourth", FakeUI())

        assert cache.lookups[-1] == ("fourth", SCOPE, "\n".join(["second", "third"][-CACHE_CONTEXT_TURNS:]))


class TestScope:
    def test_follows_the_model_in_use(
        self, make_conversation: Callable[..., Conversation], provider_manager: MagicMock
    ) -> None:
        cache = FakeCache()
        conversation = make_conversation(cache)
        conversation.chat("What is Redis?", FakeUI())
        provider_manager.provider.return_value.current_model.return_value = "llama3.2:3b"

        conversation.chat("What is Redis?", FakeUI())

        assert [scope.model for _, scope, _ in cache.lookups] == [MODEL, "llama3.2:3b"]

    def test_follows_the_active_persona(
        self, make_conversation: Callable[..., Conversation], persona_manager: MagicMock
    ) -> None:
        cache = FakeCache()
        persona_manager.active_persona.return_value = Persona(None, "analyst", "Analyses data.", "You are an analyst.")

        make_conversation(cache).chat("What is Redis?", FakeUI())

        assert cache.lookups == [("What is Redis?", CacheScope(model=MODEL, persona="analyst"), "")]
        assert cache.stored[0][1] == CacheScope(model=MODEL, persona="analyst")


class TestHistory:
    def test_hit_is_saved_to_the_database(self, make_conversation: Callable[..., Conversation], db: MagicMock) -> None:
        cache = FakeCache({("What is Redis?", ""): "From the cache."})

        make_conversation(cache).chat("What is Redis?", FakeUI())

        contents = [call.args[2] for call in db.add_message_to_branch.call_args_list]
        assert contents == ["What is Redis?", "From the cache."]


def test_conversation_delegates_generation_to_react() -> None:
    provider_manager = MagicMock()
    persona_manager = MagicMock()
    persona_manager.active_persona.return_value = None
    react = MagicMock(spec=ReAct)
    react.generate_answer.return_value = iter(
        [ChatResponse({"message": {"role": "assistant", "content": "Hello"}})],
    )
    conversation = Conversation(provider_manager, persona_manager, ChatPipeline(ModelStage(react)), EventBus())

    def record_message(role: Role, content: str, ui: object) -> bool:
        del ui
        conversation.history.append(role, content)
        return True

    conversation.add_message = MagicMock(side_effect=record_message)
    ui = MagicMock()
    ui.wait_event.return_value = False

    conversation.chat("Bonjour", ui)

    react.generate_answer.assert_called_once()
    assert react.generate_answer.call_args.args[0] == [{"role": "user", "content": "Bonjour"}]
    assert [call.args[0].content for call in ui.send_ai_response.call_args_list] == ["Hello"]
    conversation.add_message.assert_called_with(Role.ASSISTANT, "Hello", ui)
