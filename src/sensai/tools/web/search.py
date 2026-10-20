from __future__ import annotations

from typing import TYPE_CHECKING

from sensai.tools.tool import Tool, get_parameter
from sensai.tools.web._shaping import clean_text, header, truncate_tokens, unavailable
from sensai.tools.web.error import WebBackendError

if TYPE_CHECKING:
    from sensai.tools.web.backend.search import SearchBackend

MAX_RESULTS = 5
SNIPPET_TOKENS = 75
SEARCH_MAX_TOKENS = 600


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
    def description(self) -> str:
        """What the tool does, read by the model to decide when to call it."""
        return (
            f"Search the web. Returns at most {MAX_RESULTS} results as title, URL and short snippet. "
            "Call web_fetch on the most relevant URL to read the page itself."
        )

    @property
    def parameters(self) -> dict:
        """JSON schema of the arguments accepted by `run`."""
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        }

    def run(self, **kwargs: object) -> str:
        """Search the web for `query`, failures being reported to the model as text."""
        query = get_parameter("query", str, **kwargs)
        try:
            results = self._backend.search(query, MAX_RESULTS)
        except WebBackendError as err:
            return unavailable("Web search", err)
        if not results:
            return f"No web results for: {query}"
        output = "\n\n".join(
            f"{header(i, result.title, result.url)}\n{truncate_tokens(clean_text(result.snippet), SNIPPET_TOKENS)}"
            for i, result in enumerate(results[:MAX_RESULTS], start=1)
        )
        return truncate_tokens(output, SEARCH_MAX_TOKENS)
