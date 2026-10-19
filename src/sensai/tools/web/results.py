from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    """One web search hit: never more than a short snippet of the page."""

    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class Page:
    """Main text content of one fetched page, navigation and boilerplate removed."""

    title: str
    url: str
    text: str
