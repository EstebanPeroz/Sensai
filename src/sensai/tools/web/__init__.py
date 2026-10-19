from __future__ import annotations

from typing import TYPE_CHECKING

from sensai.tools.web.backend._ollama import OllamaClient
from sensai.tools.web.backend.fetch import FetchChain
from sensai.tools.web.backend.fetch.http import HttpFetch
from sensai.tools.web.backend.fetch.ollama import OllamaFetch
from sensai.tools.web.backend.search import SearchChain
from sensai.tools.web.backend.search.ddgs import DuckDuckGoSearch
from sensai.tools.web.backend.search.ollama import OllamaSearch
from sensai.tools.web.fetch import WebFetchTool
from sensai.tools.web.search import WebSearchTool

if TYPE_CHECKING:
    from sensai.config.settings import WebSettings
    from sensai.llm.embedder import Embedder
    from sensai.tools.tool import Tool

__all__ = ["WebFetchTool", "WebSearchTool", "web_tools"]


def web_tools(settings: WebSettings, embedder: Embedder) -> list[Tool]:
    """Build web_search and web_fetch: Ollama's API first, keyless backends as fallback."""
    ollama = OllamaClient(settings.api_base_url, settings.api_key, settings.timeout)
    return [
        WebSearchTool(SearchChain([OllamaSearch(ollama), DuckDuckGoSearch(settings.timeout)])),
        WebFetchTool(FetchChain([OllamaFetch(ollama), HttpFetch(settings.timeout)]), embedder),
    ]
