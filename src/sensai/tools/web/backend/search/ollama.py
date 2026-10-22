from __future__ import annotations

from typing import TYPE_CHECKING

from sensai.tools.web.backend.search import SearchBackend
from sensai.tools.web.error import WebBackendError
from sensai.tools.web.results import SearchResult

if TYPE_CHECKING:
    from sensai.tools.web.backend._ollama import OllamaClient


class OllamaSearch(SearchBackend):
    """Search through Ollama's hosted `api/web_search`."""

    name = "ollama"

    def __init__(self, client: OllamaClient) -> None:
        """Send the requests through the shared Ollama `client`."""
        self._client = client

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Search the web through `api/web_search`."""
        content = self._client.post("api/web_search", {"query": query, "max_results": max_results})
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
