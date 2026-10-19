import requests
import trafilatura

from sensai.tools.web.backend import FetchBackend
from sensai.tools.web.error import WebBackendError
from sensai.tools.web.results import Page

_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) Sensai/0.1"


class HttpFetch(FetchBackend):
    """Keyless fetch: downloads the page and extracts its main content locally with trafilatura."""

    name = "http"

    def __init__(self, timeout: float) -> None:
        """Set the download `timeout`, in seconds."""
        self._timeout = timeout

    def fetch(self, url: str) -> Page:
        """Download `url` and keep its main text, dropping navigation, footer and ads."""
        try:
            response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=self._timeout)
        except requests.RequestException as err:
            raise WebBackendError(self.name, str(err)) from None
        if not response.ok:
            raise WebBackendError(self.name, f"status {response.status_code}")

        text = trafilatura.extract(response.text, url=url, include_comments=False, include_tables=True)
        if not text:
            raise WebBackendError(self.name, "no main content found")
        metadata = trafilatura.extract_metadata(response.text, default_url=url)
        title = metadata.title if metadata is not None and metadata.title else ""
        return Page(title=title, url=url, text=text)
