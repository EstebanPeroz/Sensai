from __future__ import annotations

from typing import TYPE_CHECKING

from sensai.tools.web.backend.fetch import FetchBackend
from sensai.tools.web.error import WebBackendError
from sensai.tools.web.results import Page

if TYPE_CHECKING:
    from sensai.tools.web.backend._ollama import OllamaClient


class OllamaFetch(FetchBackend):
    """Fetch through Ollama's hosted `api/web_fetch`."""

    name = "ollama"

    def __init__(self, client: OllamaClient) -> None:
        """Send the requests through the shared Ollama `client`."""
        self._client = client

    def fetch(self, url: str) -> Page:
        """Read one page through `api/web_fetch`, which already returns its main content."""
        content = self._client.post("api/web_fetch", {"url": url})
        text = content.get("content")
        if not isinstance(text, str) or not text.strip():
            raise WebBackendError(self.name, "empty page content")
        return Page(title=str(content.get("title", "")), url=url, text=text)
