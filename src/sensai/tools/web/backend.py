from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from sensai.tools.web.error import WebBackendError

if TYPE_CHECKING:
    from sensai.tools.web.results import Page, SearchResult

logger = logging.getLogger(__name__)


class SearchBackend(ABC):
    """Strategy answering web search queries."""

    name: str

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Return at most `max_results` hits for `query`, raising WebBackendError on failure."""


class FetchBackend(ABC):
    """Strategy reading the main content of one page."""

    name: str

    @abstractmethod
    def fetch(self, url: str) -> Page:
        """Return the main content of `url`, raising WebBackendError on failure."""


class FallbackSearch(SearchBackend):
    """Asks each backend in turn, switching to the next one on error."""

    name = "fallback"

    def __init__(self, backends: list[SearchBackend]) -> None:
        """Keep the backends in priority order."""
        self._backends = backends

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Return the hits of the first backend that answers."""
        errors: list[str] = []
        for backend in self._backends:
            try:
                results = backend.search(query, max_results)
            except WebBackendError as err:
                logger.info("web_search: %s failed, trying next backend (%s)", backend.name, err.reason)
                errors.append(f"{backend.name}: {err.reason}")
                continue
            logger.info("web_search: answered by %s", backend.name)
            return results
        raise WebBackendError(self.name, "; ".join(errors) or "no backend configured")


class FallbackFetch(FetchBackend):
    """Asks each backend in turn, switching to the next one on error."""

    name = "fallback"

    def __init__(self, backends: list[FetchBackend]) -> None:
        """Keep the backends in priority order."""
        self._backends = backends

    def fetch(self, url: str) -> Page:
        """Return the page read by the first backend that answers."""
        errors: list[str] = []
        for backend in self._backends:
            try:
                page = backend.fetch(url)
            except WebBackendError as err:
                logger.info("web_fetch: %s failed, trying next backend (%s)", backend.name, err.reason)
                errors.append(f"{backend.name}: {err.reason}")
                continue
            logger.info("web_fetch: answered by %s", backend.name)
            return page
        raise WebBackendError(self.name, "; ".join(errors) or "no backend configured")
