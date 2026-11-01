import base64
import io
from email.message import Message
from unittest.mock import MagicMock

import pytest
from pypdf import PdfWriter
from pypdf.generic import ContentStream, DictionaryObject, NameObject
from sensai_protocol import ClientCommand
from sensai_server.rag import sources
from sensai_server.rag.error import RagError, SourceFetchError, UnsupportedFormatError
from sensai_server.rag.sources import collect, extract_html, extract_pdf

from tests.server.fakes import AnsweringOutbound

HTML = "<html><head><style>p {}</style></head><body><nav>menu</nav><p>Hello page</p><script>x()</script></body></html>"


def _pdf(text: str) -> bytes:
    """A one-page PDF showing `text`."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}  # noqa: SLF001
    )
    stream = ContentStream(None, writer)
    stream.set_data(f"BT /F1 12 Tf 10 150 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)  # noqa: SLF001
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _b64(data: bytes) -> dict:
    return {"content": base64.b64encode(data).decode()}


class _Response(io.BytesIO):
    """An `urlopen` response serving `data` as `content_type`."""

    def __init__(self, data: bytes, content_type: str) -> None:
        super().__init__(data)
        self.headers = Message()
        self.headers["Content-Type"] = content_type


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, data: bytes, content_type: str) -> MagicMock:
    urlopen = MagicMock(return_value=_Response(data, content_type))
    monkeypatch.setattr(sources.urllib.request, "urlopen", urlopen)
    return urlopen


class TestExtract:
    def test_html_drops_the_tags_that_are_not_content(self) -> None:
        assert extract_html(HTML) == "Hello page"

    def test_pdf_returns_the_text_of_its_pages(self) -> None:
        assert extract_pdf(_pdf("Hello PDF")) == "Hello PDF"

    def test_data_that_is_not_a_pdf_raises(self) -> None:
        with pytest.raises(SourceFetchError):
            extract_pdf(b"not a pdf at all")


class TestClientPath:
    @pytest.mark.parametrize("path", ["~/notes.txt", "~/README.md"])
    async def test_a_text_file_is_read_as_is(self, path: str) -> None:
        out = AnsweringOutbound({ClientCommand.READ_FILE: {"content": "plain text"}})
        assert await collect(path, out) == "plain text"
        assert [(r.command, r.params) for r in out.requests()] == [(ClientCommand.READ_FILE, {"path": path})]

    async def test_an_empty_file_is_not_an_error(self) -> None:
        out = AnsweringOutbound({ClientCommand.READ_FILE: {"content": ""}})
        assert await collect("empty.txt", out) == ""

    @pytest.mark.parametrize("path", ["page.html", "PAGE.HTM"])
    async def test_an_html_file_is_reduced_to_its_text(self, path: str) -> None:
        out = AnsweringOutbound({ClientCommand.READ_FILE: {"content": HTML}})
        assert await collect(path, out) == "Hello page"

    async def test_a_pdf_is_read_as_bytes(self) -> None:
        out = AnsweringOutbound({ClientCommand.READ_BINARY_FILE: _b64(_pdf("Hello PDF"))})
        assert await collect("doc.pdf", out) == "Hello PDF"
        assert [r.command for r in out.requests()] == [ClientCommand.READ_BINARY_FILE]

    @pytest.mark.parametrize("path", ["image.png", "Makefile"])
    async def test_an_unsupported_extension_asks_the_client_nothing(self, path: str) -> None:
        out = AnsweringOutbound({})
        with pytest.raises(UnsupportedFormatError):
            await collect(path, out)
        assert out.requests() == []

    async def test_a_client_error_becomes_a_fetch_error_keeping_its_reason(self) -> None:
        out = AnsweringOutbound({ClientCommand.READ_FILE: "file not found"})
        with pytest.raises(SourceFetchError, match="file not found"):
            await collect("missing.txt", out)

    async def test_a_binary_client_error_becomes_a_fetch_error(self) -> None:
        out = AnsweringOutbound({ClientCommand.READ_BINARY_FILE: "file too large"})
        with pytest.raises(SourceFetchError, match="file too large"):
            await collect("big.pdf", out)

    async def test_a_pdf_that_is_not_one_raises(self) -> None:
        out = AnsweringOutbound({ClientCommand.READ_BINARY_FILE: _b64(b"not a pdf")})
        with pytest.raises(SourceFetchError):
            await collect("fake.pdf", out)


class TestUrl:
    async def test_a_page_is_parsed_as_html(self, monkeypatch: pytest.MonkeyPatch) -> None:
        urlopen = _patch_urlopen(monkeypatch, HTML.encode(), "text/html; charset=utf-8")
        out = AnsweringOutbound({})
        assert await collect("https://example.com/page", out) == "Hello page"
        assert out.requests() == []
        assert urlopen.call_args.kwargs["timeout"] == 10

    async def test_a_pdf_content_type_is_parsed_as_pdf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_urlopen(monkeypatch, _pdf("Hello PDF"), "application/pdf")
        assert await collect("https://example.com/download", AnsweringOutbound({})) == "Hello PDF"

    async def test_a_pdf_path_is_parsed_as_pdf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_urlopen(monkeypatch, _pdf("Hello PDF"), "application/octet-stream")
        assert await collect("http://example.com/doc.PDF?v=1", AnsweringOutbound({})) == "Hello PDF"

    async def test_an_unreachable_url_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sources.urllib.request, "urlopen", MagicMock(side_effect=TimeoutError("timed out")))
        with pytest.raises(SourceFetchError, match="timed out"):
            await collect("https://example.com", AnsweringOutbound({}))

    @pytest.mark.parametrize("url", ["http://", "https://exa mple.com:port/"])
    async def test_a_malformed_url_raises(self, url: str) -> None:
        out = AnsweringOutbound({})
        with pytest.raises(SourceFetchError):
            await collect(url, out)
        assert out.requests() == []

    async def test_every_failure_is_a_rag_error(self) -> None:
        with pytest.raises(RagError):
            await collect("image.png", AnsweringOutbound({}))
