from ddgs import DDGS
from ddgs.exceptions import DDGSException

from sensai.tools.web.backend import SearchBackend
from sensai.tools.web.error import WebBackendError
from sensai.tools.web.results import SearchResult


class DuckDuckGoSearch(SearchBackend):
    """Keyless search through the `ddgs` package: unofficial scraping, may be rate limited, snippets only."""

    name = "ddgs"

    def __init__(self, timeout: float) -> None:
        """Set the per-request `timeout`, in seconds."""
        self._timeout = timeout

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Search the web through DuckDuckGo and its sibling engines."""
        try:
            results = DDGS(timeout=self._timeout).text(query, max_results=max_results)
        except DDGSException as err:
            raise WebBackendError(self.name, str(err) or type(err).__name__) from None
        return [
            SearchResult(
                title=str(result.get("title", "")),
                url=str(result.get("href", "")),
                snippet=str(result.get("body", "")),
            )
            for result in results[:max_results]
            if result.get("href")
        ]
