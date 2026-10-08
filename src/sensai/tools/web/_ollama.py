import requests

from sensai.tools.web.backend import FetchBackend, Page, SearchBackend, SearchResult, WebBackendError


class OllamaWeb(SearchBackend, FetchBackend):
    """Ollama's hosted web search and fetch API: needs an API key, queries leave the machine."""

    name = "ollama"

    def __init__(self, base_url: str, api_key: str | None, timeout: float) -> None:
        """Bind the backend to the API at `base_url`, authenticated with `api_key`."""
        self._base_url = base_url.rstrip("/") + "/"
        self._api_key = api_key
        self._timeout = timeout

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Search the web through `api/web_search`."""
        content = self._post("api/web_search", {"query": query, "max_results": max_results})
        results = content.get("results")
        if not isinstance(results, list):
            raise WebBackendError(self.name, "missing 'results' in response")
        return [
            SearchResult(
                title=str(result.get("title", "")),
                url=str(result.get("url", "")),
                snippet=str(result.get("content", "")),
            )
            for result in results[:max_results]
            if isinstance(result, dict) and result.get("url")
        ]

    def fetch(self, url: str) -> Page:
        """Read one page through `api/web_fetch`, which already returns its main content."""
        content = self._post("api/web_fetch", {"url": url})
        text = content.get("content")
        if not isinstance(text, str) or not text.strip():
            raise WebBackendError(self.name, "empty page content")
        return Page(title=str(content.get("title", "")), url=url, text=text)

    def _post(self, endpoint: str, payload: dict) -> dict:
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
