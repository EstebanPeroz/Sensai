from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal, overload

import requests

from sensai.llm.adapter import InterfaceAdapter
from sensai.llm.responses import ChatResponse, ShowResponse

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sensai.config.settings import OllamaSettings


class OllamaAdapter(InterfaceAdapter):
    """Ollama API wrapper to call needed endpoints."""

    def __init__(self, settings: OllamaSettings) -> None:
        """Init ollama Adapter."""
        super().__init__()
        self._base_url = settings.base_url.rstrip("/") + "/"
        self._embedding_model = settings.embedding_model
        self._timeout = settings.timeout

    @overload
    def chat(self, payload: dict, *, stream: Literal[True]) -> Iterator[ChatResponse] | None: ...
    @overload
    def chat(self, payload: dict, *, stream: Literal[False]) -> ChatResponse | None: ...

    def chat(self, payload: dict, *, stream: bool) -> ChatResponse | Iterator[ChatResponse] | None:
        """Send a chat call on the ollama API."""
        payload["stream"] = stream
        if stream:
            chunks = self._call_stream("api/chat", payload=payload)
            if chunks is None:
                return None
            return (ChatResponse(chunk) for chunk in chunks)
        content = self._call("api/chat", payload=payload)
        if content is None:
            return None
        return ChatResponse(content)

    def embedding(self, message: str) -> list:
        """Send a message to embed and receive a list of vector."""
        content = self.embeddings(messages=[message])
        if content == []:
            return content
        return content[0]

    def embeddings(self, messages: list[str]) -> list[list]:
        """Send messages to embed and receive a list of vector for each message."""
        content = self._call(
            endpoint="api/generate",
            payload={"model": self._embedding_model, "inputs": messages},
        )

        if content is None:
            return []
        return content.get("embeddings", [])

    def show(self, model_name: str) -> ShowResponse | None:
        """Get info on a specified model on the ollama API."""
        content = self._call("api/show", {"model": model_name})
        if content is None:
            return None
        response: ShowResponse = ShowResponse(model=model_name)

        capabilities = content.get("capabilities", "")
        if capabilities == "":
            return response
        if "tools" in capabilities:
            response.tools = True
        if "thinking" in capabilities:
            response.think = True
        return response

    def load(self, model_name: str) -> bool:
        """Load a model with the ollama API."""
        content = self._call("api/generate", {"model": model_name})
        if content is None:
            return False
        return content.get("done", False)

    def unload(self, model_name: str) -> bool:
        """Load a model with the ollama API."""
        content = self._call("api/generate", {"model": model_name, "keep_alive": 0})
        if content is None:
            return False
        return content.get("done", False)

    def _call(self, endpoint: str, payload: dict) -> dict | None:
        try:
            result = requests.post(
                self._base_url + endpoint,
                json=payload,
                timeout=self._timeout,
            )
        except requests.RequestException:
            return None

        if not result.ok:
            return None

        try:
            return result.json()
        except requests.exceptions.JSONDecodeError:
            return None

    def _call_stream(self, endpoint: str, payload: dict) -> Iterator[dict] | None:
        try:
            result = requests.post(
                self._base_url + endpoint,
                json=payload,
                stream=True,
                timeout=self._timeout,
            )
        except requests.RequestException:
            return None

        if not result.ok:
            return None

        return self._iter_json_lines(result)

    @staticmethod
    def _iter_json_lines(result: requests.Response) -> Iterator[dict]:
        try:
            for line in result.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        except requests.RequestException:
            return
        finally:
            result.close()
