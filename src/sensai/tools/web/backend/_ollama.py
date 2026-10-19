import requests

from sensai.tools.web.error import WebBackendError


class OllamaClient:
    """Ollama's hosted web API, shared by its search and fetch backends: needs an API key, queries leave the machine."""

    name = "ollama"

    def __init__(self, base_url: str, api_key: str | None, timeout: float) -> None:
        """Bind the client to the API at `base_url`, authenticated with `api_key`."""
        self._base_url = base_url.rstrip("/") + "/"
        self._api_key = api_key
        self._timeout = timeout

    def post(self, endpoint: str, payload: dict) -> dict:
        """POST `payload` to `endpoint` and return the decoded JSON object, raising WebBackendError on failure."""
        if not self._api_key:
            raise WebBackendError(self.name, "OLLAMA_API_KEY is not set")
        try:
            response = requests.post(
                self._base_url + endpoint,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
        except requests.RequestException as err:
            raise WebBackendError(self.name, str(err)) from None
        if not response.ok:
            raise WebBackendError(self.name, f"status {response.status_code}")
        try:
            content = response.json()
        except requests.exceptions.JSONDecodeError:
            raise WebBackendError(self.name, "invalid JSON response") from None
        if not isinstance(content, dict):
            raise WebBackendError(self.name, "unexpected response shape")
        return content
