from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal, overload

import requests

from sensai.llm.adapter import ProviderAdapter
from sensai.llm.responses import ChatResponse, ShowResponse

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sensai.config.settings import OllamaSettings


class OllamaAdapter(ProviderAdapter):
    """Ollama API wrapper to call needed endpoints."""

    def __init__(self, settings: OllamaSettings) -> None:
        """Init ollama Adapter."""
        super().__init__()
        self._base_url: str = settings.base_url.rstrip("/") + "/"

    @overload
    def chat(self, payload: dict, *, stream: Literal[True]) -> Iterator[ChatResponse] | None: ...
    @overload
    def chat(self, payload: dict, *, stream: Literal[False]) -> ChatResponse | None: ...

    def chat(self, payload: dict, *, stream: bool) -> ChatResponse | Iterator[ChatResponse] | None:
        """Send a chat call on the ollama API."""
        payload["stream"] = stream
        if stream:
            chunks = self._call_stream("api/chat", payload=payload, timeout=30)
            if chunks is None:
                return None
            return (ChatResponse(chunk) for chunk in chunks)
        content = self._call("api/chat", payload=payload, timeout=10)
        if content is None:
            return None
        return ChatResponse(content)

    def embedding(self, message: str, model: str) -> list:
        """Send a message to embed with model and receive a list of vector."""
        content = self.embeddings(messages=[message], model=model)
        if content == []:
            return content
        return content[0]

    def embeddings(self, messages: list[str], model: str) -> list[list]:
        """Send messages to embed with model and receive a list of vector for each message."""
        content = self._call(
            endpoint="api/embed",
            payload={"model": model, "input": messages},
            timeout=5,
        )

        if content is None:
            return []
        return content.get("embeddings", [])

    def available_models(self) -> list[str]:
        """List the models installed on the Ollama server."""
        content = self._get("api/tags", timeout=3)
        if content is None:
            return []
        return [model["name"] for model in content.get("models", []) if "name" in model]

    def get_default_model(self) -> str | None:
        """Return the first model installed on the Ollama server, if any."""
        models = self.available_models()
        if not models:
            return None
        return models[0]

    def show(self, model_name: str) -> ShowResponse | None:
        """Get info on a specified model on the ollama API."""
        content = self._call("api/show", {"model": model_name}, timeout=3)
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

    def _call(self, endpoint: str, payload: dict, *, timeout: float = 10) -> dict | None:
        try:
            result = requests.post(
                self._base_url + endpoint,
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException:
            return None
        return self._json(result)

    def _get(self, endpoint: str, *, timeout: float = 10) -> dict | None:
        try:
            result = requests.get(self._base_url + endpoint, timeout=timeout)
        except requests.RequestException:
            return None
        return self._json(result)

    @staticmethod
    def _json(result: requests.Response) -> dict | None:
        if not result.ok:
            return None

        try:
            return result.json()
        except requests.exceptions.JSONDecodeError:
            return None

    def _call_stream(self, endpoint: str, payload: dict, *, timeout: float = 10) -> Iterator[dict] | None:
        try:
            result = requests.post(
                self._base_url + endpoint,
                json=payload,
                stream=True,
                timeout=timeout,
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
