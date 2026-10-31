import threading
from collections.abc import AsyncGenerator, Iterator
from typing import ClassVar, Literal, cast, overload, override

from fastapi import WebSocket
from sensai_protocol import ClientCommand, CommandContext, RequestCommand, ServerMessage, parse_server_message
from sensai_server.app.outbound import Outbound
from sensai_server.cache import CacheError
from sensai_server.config.settings import OllamaSettings
from sensai_server.llm.embedder import Embedder
from sensai_server.llm.error import EmbeddingError, RequestCallError
from sensai_server.llm.provider import ChatResponse, ProviderAdapter, ShowResponse
from sensai_server.pipeline import ChatContext


class RecordingOutbound(Outbound):
    """An `Outbound` that records the messages it sends instead of writing to a socket."""

    def __init__(self) -> None:
        super().__init__(cast("WebSocket", None))
        self.sent: list[ServerMessage] = []

    async def send(self, message: ServerMessage) -> bool:
        self.sent.append(parse_server_message(message.model_dump_json()))
        return True


class AnsweringOutbound(RecordingOutbound):
    """A `RecordingOutbound` whose client runs client commands: each `request_command` gets the canned answer.

    `answers` maps a command to the `ctx` the client answers with, or to a `str`, the `error` it fails with.
    """

    def __init__(self, answers: dict[ClientCommand, dict | str]) -> None:
        super().__init__()
        self.answers = answers

    async def send(self, message: ServerMessage) -> bool:
        await super().send(message)
        if isinstance(message, RequestCommand):
            answer = self.answers[message.command]
            ctx, error = (None, answer) if isinstance(answer, str) else (answer, None)
            self.resolve(
                message.request_id,
                CommandContext(request_id=message.request_id, command=message.command, ctx=ctx, error=error),
            )
        return True

    def requests(self) -> list[RequestCommand]:
        """The client commands asked so far."""
        return [message for message in self.sent if isinstance(message, RequestCommand)]


class FakeAdapter(ProviderAdapter):
    """A provider serving `models` without any network call; `load` can be made to refuse or fail.

    A streamed `chat` records its payload in `payloads` and `last_payload`. It yields the next stream of `script`,
    one per call (think, act, observe...), each a list of Ollama payloads; once `script` runs out, it yields
    `replies`. When `block` is set, it waits on it after the first reply. The `closed` event is set once a stream
    has been closed. `show` records the model in `shown` and reports `tools` as its tool capability.
    """

    models: ClassVar[list[str]] = ["a", "b"]
    refused: ClassVar[set[str]] = set()
    failing: ClassVar[set[str]] = set()
    unreachable: ClassVar[bool] = False
    replies: ClassVar[list[dict]] = []
    script: ClassVar[list[list[dict]]] = []
    payloads: ClassVar[list[dict]] = []
    last_payload: ClassVar[dict | None] = None
    tools: ClassVar[bool] = False
    shown: ClassVar[list[str]] = []
    block: ClassVar[threading.Event | None] = None
    closed: ClassVar[threading.Event] = threading.Event()

    def __init__(self, settings: OllamaSettings) -> None:
        self.settings = settings
        self.calls: list[tuple[str, str]] = []

    @overload
    def chat(self, payload: dict, *, stream: Literal[True]) -> Iterator[ChatResponse]: ...
    @overload
    def chat(self, payload: dict, *, stream: Literal[False]) -> ChatResponse: ...

    @override
    def chat(self, payload: dict, *, stream: bool) -> ChatResponse | Iterator[ChatResponse]:
        if not stream:
            raise NotImplementedError
        type(self).last_payload = payload
        self.payloads.append(payload)
        replies = self.script.pop(0) if self.script else self.replies
        return self._stream(replies)

    def _stream(self, replies: list[dict]) -> Iterator[ChatResponse]:
        try:
            for index, reply in enumerate(replies):
                if index == 1 and self.block is not None:
                    self.block.wait()
                yield ChatResponse(reply)
        finally:
            self.closed.set()

    @override
    def embedding(self, message: str, model: str) -> list:
        raise NotImplementedError

    @override
    def embeddings(self, messages: list[str], model: str) -> list[list]:
        raise NotImplementedError

    @override
    def show(self, model_name: str) -> ShowResponse:
        self.shown.append(model_name)
        return ShowResponse(model_name, tools=self.tools)

    @override
    def load(self, model_name: str) -> bool:
        self.calls.append(("load", model_name))
        if model_name in self.failing:
            raise RequestCallError(model_name)
        if model_name in self.refused:
            return False
        self._current_model = model_name
        return True

    @override
    def unload(self, model_name: str) -> bool:
        raise NotImplementedError

    @override
    def list(self) -> list[str]:
        if self.unreachable:
            msg = " Get -> connection refused"
            raise RequestCallError(msg)
        return self.models


class FakeEmbedder(Embedder):
    """An `Embedder` whose vectors are deterministic and need no provider; it records every batch in `batches`.

    Texts mentioning "engine" land on one axis, the others on the opposite one, with a small drift per position so
    that consecutive scores are never exactly equal: an adaptive threshold sitting right on a score would find no
    breakpoint at all. When `failing` is set, `embed` raises `EmbeddingError`, as the real one does.
    """

    def __init__(self, *, failing: bool = False) -> None:
        super().__init__(FakeAdapter(OllamaSettings()), "fake-embed")
        self.batches: list[list[str]] = []
        self.failing = failing

    @override
    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        if self.failing:
            msg = f"embedding model '{self.model}' is not available"
            raise EmbeddingError(msg)
        return [
            [0.1 + i * 0.01, 1.0] if "engine" in text.lower() else [1.0, 0.1 + i * 0.01] for i, text in enumerate(texts)
        ]


def chunk(content: str, *, done: bool = False) -> dict:
    """An Ollama `/api/chat` payload streaming `content`."""
    return {"done": done, "message": {"role": "assistant", "content": content}}


def tool_call_chunk(*calls: dict) -> dict:
    """An Ollama `/api/chat` payload calling `calls`, sent before the `done` chunk as Ollama does."""
    return {"done": False, "message": {"role": "assistant", "content": "", "tool_calls": list(calls)}}


def call(name: str, arguments: dict | None = None) -> dict:
    """An LLM-style call to the tool `name`."""
    return {"function": {"name": name, "arguments": arguments or {}}}


class RecordingNext:
    """A `Next` that streams `chunks` and records the context of every call."""

    def __init__(self, *chunks: ChatResponse) -> None:
        self.chunks = chunks
        self.calls: list[ChatContext] = []
        self.seen_messages: list[list[dict]] = []

    async def __call__(self, context: ChatContext) -> AsyncGenerator[ChatResponse]:
        self.calls.append(context)
        self.seen_messages.append(list(context.messages))
        for item in self.chunks:
            yield item


class FakeSessionCache:
    """A `SessionCache` held in memory, matching prompts exactly; each operation can be made to fail.

    It records the prompts looked up in `lookups`, the `(prompt, answer)` pairs stored in `stored` and the number of
    `clears`. A failing operation raises `CacheError`, as the real one does when Redis fails, or `lookup_error` when
    given.
    """

    def __init__(
        self,
        answers: dict[str, str] | None = None,
        *,
        lookup_error: Exception | None = None,
        failing_store: bool = False,
        failing_clear: bool = False,
    ) -> None:
        self.answers = dict(answers or {})
        self.lookup_error = lookup_error
        self.failing_store = failing_store
        self.failing_clear = failing_clear
        self.lookups: list[str] = []
        self.stored: list[tuple[str, str]] = []
        self.clears = 0

    async def lookup(self, prompt: str) -> str | None:
        self.lookups.append(prompt)
        if self.lookup_error is not None:
            raise self.lookup_error
        return self.answers.get(prompt)

    async def store(self, prompt: str, answer: str) -> None:
        if self.failing_store:
            msg = "the Redis server does not answer"
            raise CacheError(msg)
        self.stored.append((prompt, answer))
        self.answers[prompt] = answer

    async def clear(self) -> None:
        if self.failing_clear:
            msg = "the Redis server does not answer"
            raise CacheError(msg)
        self.clears += 1
        self.answers.clear()
