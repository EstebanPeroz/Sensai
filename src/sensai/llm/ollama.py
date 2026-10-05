from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal, overload

import requests

from sensai.error import SensaiError
from sensai.llm.adapter import ProviderAdapter
from sensai.llm.error import JsonError, RequestCallError, RequestStatusError
from sensai.llm.responses import ChatResponse, ShowResponse

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sensai.config.settings import ProviderSettings


class OllamaAdapter(ProviderAdapter):
    """Ollama API wrapper to call needed endpoints."""

    def __init__(self, settings: ProviderSettings) -> None:
        """Init ollama Adapter with its connection settings and the embedding model, if one is configured."""
        super().__init__()
        self._base_url: str = settings.base_url.rstrip("/") + "/"

    @overload
    def chat(self, payload: dict, *, stream: Literal[True]) -> Iterator[ChatResponse]: ...
    @overload
    def chat(self, payload: dict, *, stream: Literal[False]) -> ChatResponse: ...

    def chat(self, payload: dict, *, stream: bool) -> ChatResponse | Iterator[ChatResponse]:
        """Send a chat call on the ollama API."""
        payload["stream"] = stream
        if stream:
            chunks = self._post_stream("api/chat", payload=payload, timeout=30)
            return (ChatResponse(chunk) for chunk in chunks)
        content = self._post("api/chat", payload=payload, timeout=10)
        return ChatResponse(content)

    def embedding(self, message: str, model: str) -> list:
        """Send a message to embed and receive a list of vector."""
        content = self.embeddings(messages=[message], model=model)
        if content == []:
            return content
        return content[0]

    def embeddings(self, messages: list[str], model: str) -> list[list]:
        """Send messages to embed and receive a list of vector for each message, none without an embedding model."""
        try:
            self.show(model)
        except SensaiError:
            return []
        content = self._post(
            endpoint="api/embed",
            payload={"model": model, "input": messages},
            timeout=5,
        )

        return content.get("embeddings", [])

    def show(self, model_name: str) -> ShowResponse:
        """Get info on a specified model on the ollama API."""
        content = self._post("api/show", {"model": model_name}, timeout=3)
        response: ShowResponse = ShowResponse(model=model_name)

        capabilities: list = content.get("capabilities", [])
        if capabilities == []:
            return response
        if "tools" in capabilities:
            response.tools = True
        if "thinking" in capabilities:
            response.think = True
        return response

    def load(self, model_name: str) -> bool:
        """Load a model with the ollama API."""
        content = self._post("api/generate", {"model": model_name})
        return content.get("done", False)

    def unload(self, model_name: str) -> bool:
        """Load a model with the ollama API."""
        content = self._post("api/generate", {"model": model_name, "keep_alive": 0})
        return content.get("done", False)

    def list(self) -> list[str]:
        """List of model given by the provider."""
        content = self._get("api/tags")

        models_response: list[dict] = content.get("models", [])
        if models_response == []:
            return []

        models: list[str] = []
        for model in models_response:
            name = model.get("name", "")
            if name != "":
                models.append(name)

        return models

    def _get(self, endpoint: str, *, timeout: int = 10) -> dict:
        try:
            result = requests.get(
                self._base_url + endpoint,
                timeout=timeout,
            )
        except requests.RequestException as err:
            raise RequestCallError(" Get -> " + str(err)) from None
        return self._response_to_json(result)

    def _post(self, endpoint: str, payload: dict, *, timeout: int = 10) -> dict:
        try:
            result = requests.post(
                self._base_url + endpoint,
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as err:
            raise RequestCallError(" Post -> " + str(err)) from None
        return self._response_to_json(result)

    def _response_to_json(self, response: requests.Response) -> dict:
        if not response.ok:
            response.close()
            raise RequestStatusError(response.status_code)

        try:
            return response.json()
        except requests.exceptions.JSONDecodeError:
            raise JsonError from None

    def _post_stream(self, endpoint: str, payload: dict, *, timeout: int = 10) -> Iterator[dict]:
        try:
            result = requests.post(
                self._base_url + endpoint,
                json=payload,
                stream=True,
                timeout=timeout,
            )
        except requests.RequestException as err:
            raise RequestCallError(" Post -> " + str(err)) from None

        if not result.ok:
            result.close()
            raise RequestStatusError(result.status_code)

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
        except requests.RequestException as err:
            raise RequestCallError(" fail to get stream chunk" + str(err)) from None
        finally:
            result.close()
