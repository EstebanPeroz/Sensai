"""Turn a source, a URL or a path on the client's machine, into plain text.

The server never opens a path on its own disk: every path is the client's, read through a client command.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import urllib.request
from enum import StrEnum
from http.client import HTTPException
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, cast
from urllib.parse import urlparse

import pypdf
from bs4 import BeautifulSoup
from pypdf.errors import DependencyError, PyPdfError
from sensai_protocol import BinaryFileContent, ClientCommand, FileContent, ReadFileParams

from sensai_server.app.outbound import ClientCommandError
from sensai_server.rag.error import SourceFetchError, UnsupportedFormatError

if TYPE_CHECKING:
    from sensai_server.app.outbound import Outbound

_URL_TIMEOUT_SECONDS = 10
_DROPPED_TAGS = ["script", "style", "nav", "footer", "header", "noscript"]


class SourceKind(StrEnum):
    """How the text of a source is extracted."""

    TEXT = "text"
    HTML = "html"
    PDF = "pdf"


READERS: dict[str, SourceKind] = {
    ".txt": SourceKind.TEXT,
    ".md": SourceKind.TEXT,
    ".html": SourceKind.HTML,
    ".htm": SourceKind.HTML,
    ".pdf": SourceKind.PDF,
}


async def collect(source: str, out: Outbound) -> str:
    """Return the text of `source`: a URL fetched by the server, or a path read on the client through `out`.

    Raises:
        UnsupportedFormatError: If a client path has an extension no reader handles. The client is asked nothing.
        SourceFetchError: If the source cannot be fetched, read or decoded.

    """
    if source.startswith(("http://", "https://")):
        return await asyncio.to_thread(_collect_url, source)
    return await _collect_client_file(source, out)


def extract_html(text: str) -> str:
    """Remove the tags that are not content (<script>, <style>, <nav>...) and return the text of the page."""
    soup = BeautifulSoup(text, "html.parser")
    for element in soup(_DROPPED_TAGS):
        element.decompose()
    return soup.get_text(separator="\n", strip=True)


def extract_pdf(data: bytes) -> str:
    """Extract and join the text of every page of the PDF `data`.

    Raises:
        SourceFetchError: If `data` is not a readable PDF.

    """
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages_text = [page.extract_text() for page in reader.pages]
    except (OSError, PyPdfError, DependencyError) as error:
        msg = f"not a readable PDF: {error}"
        raise SourceFetchError(msg) from error
    return "\n".join(text for text in pages_text if text)


def _collect_url(source: str) -> str:
    """Fetch `source` over HTTP and return its text, parsed as a PDF or as HTML. Blocking: run it in a thread.

    The read is bounded: a server that accepts the connection then sends nothing raises a TimeoutError, an OSError.

    Raises:
        SourceFetchError: If the URL cannot be reached, read, or answers too slowly, or is not a readable PDF.

    """
    # S310: the caller has already checked that `source` starts with http:// or https://.
    request = urllib.request.Request(source, headers={"User-Agent": "Mozilla/5.0"})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=_URL_TIMEOUT_SECONDS) as response:  # noqa: S310
            content_type = response.headers.get_content_type()
            data = response.read()
    except (OSError, HTTPException, ValueError) as error:
        msg = f"could not fetch {source}: {error}"
        raise SourceFetchError(msg) from error
    if content_type == "application/pdf" or urlparse(source).path.lower().endswith(".pdf"):
        return extract_pdf(data)
    return extract_html(data.decode("utf-8", errors="ignore"))


async def _collect_client_file(path: str, out: Outbound) -> str:
    """Read the file at `path` on the client through `out` and return its text.

    Raises:
        UnsupportedFormatError: If the extension of `path` has no reader, before asking the client anything.
        SourceFetchError: If the client cannot read the file, or it is not a readable PDF.

    """
    ext = PurePosixPath(path).suffix.lower()
    kind = READERS.get(ext)
    if kind is None:
        msg = f"unsupported format: {ext or '(no extension)'}"
        raise UnsupportedFormatError(msg)
    if kind is SourceKind.PDF:
        data = await _read_binary(path, out)
        return await asyncio.to_thread(extract_pdf, data)
    text = await _read_text(path, out)
    if kind is SourceKind.HTML:
        return await asyncio.to_thread(extract_html, text)
    return text


async def _read_text(path: str, out: Outbound) -> str:
    """Ask the client for the text of the file at `path`.

    Raises:
        SourceFetchError: If the client does not read it, keeping the client's reason.

    """
    try:
        file = await out.request_command(ClientCommand.READ_FILE, ReadFileParams(path=path))
    except ClientCommandError as error:
        msg = f"could not read {path}: {error.reason}"
        raise SourceFetchError(msg) from error
    return cast("FileContent", file).content


async def _read_binary(path: str, out: Outbound) -> bytes:
    """Ask the client for the bytes of the file at `path`.

    Raises:
        SourceFetchError: If the client does not read it, keeping the client's reason, or sends invalid base64.

    """
    try:
        file = await out.request_command(ClientCommand.READ_BINARY_FILE, ReadFileParams(path=path))
    except ClientCommandError as error:
        msg = f"could not read {path}: {error.reason}"
        raise SourceFetchError(msg) from error
    try:
        return base64.b64decode(cast("BinaryFileContent", file).content, validate=True)
    except binascii.Error as error:
        msg = f"could not decode {path}: {error}"
        raise SourceFetchError(msg) from error
