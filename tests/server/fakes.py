import threading
from collections.abc import AsyncGenerator, Iterator
from typing import ClassVar, Literal, cast, overload, override

from fastapi import WebSocket
from sensai_protocol import ServerMessage, parse_server_message
from sensai_server.app.outbound import Outbound
from sensai_server.config.settings import OllamaSettings
from sensai_server.llm.error import RequestCallError
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


class FakeAdapter(ProviderAdapter):
    """A provider serving `models` without any network call; `load` can be made to refuse or fail.

    A streamed `chat` records its payload in `last_payload` and yields `replies`, each an Ollama payload. When
    `block` is set, it waits on it after the first reply. The `closed` event is set once the stream has been closed.
    """

    models: ClassVar[list[str]] = ["a", "b"]
    refused: ClassVar[set[str]] = set()
    failing: ClassVar[set[str]] = set()
    unreachable: ClassVar[bool] = False
    replies: ClassVar[list[dict]] = []
    last_payload: ClassVar[dict | None] = None
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
        return self._stream()

    def _stream(self) -> Iterator[ChatResponse]:
        try:
            for index, reply in enumerate(self.replies):
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
        raise NotImplementedError

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


def chunk(content: str, *, done: bool = False) -> dict:
    """An Ollama `/api/chat` payload streaming `content`."""
    return {"done": done, "message": {"role": "assistant", "content": content}}


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
