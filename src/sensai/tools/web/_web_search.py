from __future__ import annotations

from typing import TYPE_CHECKING

from sensai.tools.tool import Tool, get_parameter
from sensai.tools.web._ddgs import DuckDuckGoSearch
from sensai.tools.web._http import HttpFetch
from sensai.tools.web._ollama import OllamaWeb
from sensai.tools.web._shaping import chunk_words, clean_text, top_chunks, truncate_tokens
from sensai.tools.web.backend import FetchChain, SearchChain
from sensai.tools.web.error import WebBackendError

if TYPE_CHECKING:
    from sensai.config.settings import WebSettings
    from sensai.llm.embedder import Embedder
    from sensai.tools.web.backend import FetchBackend, SearchBackend

MAX_RESULTS = 5
SNIPPET_TOKENS = 75
SEARCH_MAX_TOKENS = 600
CHUNK_WORDS = 150
KEPT_CHUNKS = 4
MAX_TOKENS = 1500


def _unavailable(action: str, err: WebBackendError) -> str:
    return (
        f"{action} unavailable ({err.reason}). "
        "Tell the user the information could not be verified online instead of guessing."
    )


def _header(index: int, title: str, url: str) -> str:
    return f"[{index}] {title or url} | {url}"


class WebSearchTool(Tool[str]):
    """Lists the pages matching a query, as title, URL and short snippet only."""

    def __init__(self, backend: SearchBackend) -> None:
        """Search through `backend`."""
        self._backend = backend

    @property
    def name(self) -> str:
        """Name the model calls this tool by."""
        return "web_search"

    @property
    def tool_rule(self) -> dict:
        """Function schema advertised to the model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    f"Search the web. Returns at most {MAX_RESULTS} results as title, URL and short snippet. "
                    "Call web_fetch on the most relevant URL to read the page itself."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The search query."}},
                    "required": ["query"],
                },
            },
        }

    def run(self, **kwargs: object) -> str:
        """Search the web for `query`, failures being reported to the model as text."""
        query = get_parameter("query", str, **kwargs)
        try:
            results = self._backend.search(query, MAX_RESULTS)
        except WebBackendError as err:
            return _unavailable("Web search", err)
        if not results:
            return f"No web results for: {query}"
        output = "\n\n".join(
            f"{_header(i, result.title, result.url)}\n{truncate_tokens(clean_text(result.snippet), SNIPPET_TOKENS)}"
            for i, result in enumerate(results[:MAX_RESULTS], start=1)
        )
        return truncate_tokens(output, SEARCH_MAX_TOKENS)


class WebFetchTool(Tool[str]):
    """Reads one page and keeps only the passages relevant to a query."""

    def __init__(self, backend: FetchBackend, embedder: Embedder) -> None:
        """Fetch through `backend`, ranking passages with the shared `embedder`."""
        self._backend = backend
        self._embedder = embedder

    @property
    def name(self) -> str:
        """Name the model calls this tool by."""
        return "web_fetch"

    @property
    def tool_rule(self) -> dict:
        """Function schema advertised to the model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Read one web page, usually a URL returned by web_search. "
                    "Returns the passages of the page most relevant to the query, with the page title and URL "
                    "to cite."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The http(s) URL of the page to read."},
                        "query": {"type": "string", "description": "What to look for in the page."},
                    },
                    "required": ["url", "query"],
                },
            },
        }

    def run(self, **kwargs: object) -> str:
        """Fetch `url` and return its passages closest to `query`, failures being reported to the model as text."""
        url = get_parameter("url", str, **kwargs)
        query = get_parameter("query", str, **kwargs)
        try:
            page = self._backend.fetch(url)
        except WebBackendError as err:
            return _unavailable("Web fetch", err)
        passages = top_chunks(chunk_words(clean_text(page.text), CHUNK_WORDS), query, self._embedder, KEPT_CHUNKS)
        result = _header(1, page.title, page.url) + "\n\n" + "\n\n".join(passages)
        return truncate_tokens(result, MAX_TOKENS)


def web_tools(settings: WebSettings, api_key: str | None, embedder: Embedder) -> list[Tool]:
    """Build web_search and web_fetch: Ollama's API first, keyless backends as fallback."""
    ollama = OllamaWeb(settings.api_base_url, api_key, settings.timeout)
    return [
        WebSearchTool(SearchChain([ollama, DuckDuckGoSearch(settings.timeout)])),
        WebFetchTool(FetchChain([ollama, HttpFetch(settings.timeout)]), embedder),
    ]
