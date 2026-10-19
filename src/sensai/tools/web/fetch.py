from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sensai.llm.error import EmbeddingError
from sensai.tools.tool import Tool, get_parameter
from sensai.tools.web._shaping import chunk_words, clean_text, header, top_chunks, truncate_tokens, unavailable
from sensai.tools.web.error import WebBackendError

if TYPE_CHECKING:
    from sensai.llm.embedder import Embedder
    from sensai.tools.web.backend.fetch import FetchBackend

CHUNK_WORDS = 150
KEPT_CHUNKS = 4
MAX_TOKENS = 1500

logger = logging.getLogger(__name__)


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
            return unavailable("Web fetch", err)
        chunks = chunk_words(clean_text(page.text), CHUNK_WORDS)
        lines = [header(1, page.title, page.url)]
        try:
            passages = top_chunks(chunks, query, self._embedder, KEPT_CHUNKS)
        except EmbeddingError as err:
            logger.warning("web_fetch: passages not ranked (%s)", err)
            lines.append(f"(Passages not ranked by relevance, first ones kept: {err})")
            passages = chunks[:KEPT_CHUNKS]
        result = "\n\n".join([*lines, *passages])
        return truncate_tokens(result, MAX_TOKENS)
