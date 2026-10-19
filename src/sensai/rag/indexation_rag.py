import math
import re
import statistics
import unicodedata
import urllib.request
from enum import StrEnum
from http.client import HTTPException
from pathlib import Path
from typing import Protocol

import pypdf
from bs4 import BeautifulSoup
from pypdf.errors import DependencyError, PyPdfError

from sensai.error import SensaiError
from sensai.rag.error import (
    EmbeddingError,
    SourceFetchError,
    SourceNotFoundError,
    UnsupportedFormatError,
)

DEFAULT_EMBEDDING_MODEL = "bge-m3"

_MAX_WORD_CHUNKS = 50
_URL_TIMEOUT_SECONDS = 10
# The provider bounds how long one embedding request may take. Measured on
# 50-word chunks, 32 per request already overran it while 8 stayed under a third
# of the budget, and the total time does not depend on the request size anyway.
_EMBED_BATCH_SIZE = 8
_OVERLAP_SENTENCES = 1
_HTML_TAGS = re.compile(r"<[^>]+>")
# Zero-width (200b-200f), direction marks (202a-202e),
# BOM (feff), soft hyphen (00ad).
_INVISIBLE_CHARS = re.compile("[\u200b-\u200f\u202a-\u202e\ufeff\xad]")
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ])")
_SENTENCE_MAX_NB = 2


class Embedder(Protocol):
    """The only thing the indexation needs from a provider: turning texts into vectors."""

    def embeddings(self, messages: list[str], model: str) -> list[list]:
        """Return one vector per message, or an empty list when `model` is unavailable.

        Raises:
            SensaiError: If the request could not be sent, was refused, or came
                back unreadable. Anything else is a bug, and is left to surface.

        """
        ...


class ChunkStrategy(StrEnum):
    """How a document is cut into chunks."""

    AUTO = "auto"
    SENTENCE = "sentence"
    MARKDOWN = "markdown"
    SEMANTIC = "semantic"


class Indexation:
    """Turn a document into clean, embedded chunks ready to be stored."""

    _embedder: Embedder
    _model: str

    def __init__(self, embedder: Embedder, model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """Keep the `embedder` turning texts into vectors, and the `model` it must use."""
        self._embedder = embedder
        self._model = model

    def _collect_file(self, source: str) -> str:
        """Extract text content from .txt, .md, .pdf, .html files, or Web URLs.

        Raises:
            SourceNotFoundError: If `source` is not a URL and no such file exists.
            UnsupportedFormatError: If the file extension is not handled.
            SourceFetchError: If the source exists but cannot be fetched or decoded.

        """
        if source.startswith(("http://", "https://")):
            return self._collect_url(source)
        path = Path(source)
        if not path.exists():
            msg = f"no such file: {source}"
            raise SourceNotFoundError(msg)
        ext = path.suffix.lower()
        if ext in (".txt", ".md"):
            return self._read_text(path)
        if ext in (".html", ".htm"):
            return self._extract_text_from_html(self._read_text(path))
        if ext == ".pdf":
            return self._read_pdf(path)
        msg = f"unsupported format: {ext or '(no extension)'}"
        raise UnsupportedFormatError(msg)

    def _collect_url(self, source: str) -> str:
        """Fetch `source` over HTTP and return the text of the page.

        A server that accepts the connection and then sends nothing would block
        the read forever, and with it the thread dispatching the UI events, so
        the read is bounded: the TimeoutError it raises is an OSError, which the
        clause below already turns into a SourceFetchError.

        Raises:
            SourceFetchError: If the URL cannot be reached, read, or answers too slowly.

        """
        # S310: the caller has already checked that `source` starts with http:// or https://.
        request = urllib.request.Request(source, headers={"User-Agent": "Mozilla/5.0"})  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=_URL_TIMEOUT_SECONDS) as response:  # noqa: S310
                html_content = response.read().decode("utf-8", errors="ignore")
        except (OSError, HTTPException, ValueError) as error:
            msg = f"could not fetch {source}: {error}"
            raise SourceFetchError(msg) from error
        return self._extract_text_from_html(html_content)

    def _read_text(self, source: Path) -> str:
        """Read a text file as UTF-8.

        Raises:
            SourceFetchError: If the file cannot be read or decoded.

        """
        try:
            return source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            msg = f"could not read {source}: {error}"
            raise SourceFetchError(msg) from error

    def _read_pdf(self, source: Path) -> str:
        """Extract and join the text of every page of a PDF.

        Raises:
            SourceFetchError: If the file is not a readable PDF.

        """
        try:
            reader = pypdf.PdfReader(source)
            pages_text = [page.extract_text() for page in reader.pages]
        except (OSError, PyPdfError, DependencyError) as error:
            msg = f"could not read {source}: {error}"
            raise SourceFetchError(msg) from error
        return "\n".join(text for text in pages_text if text)

    def _extract_text_from_html(self, html_content: str) -> str:
        """Remove unwanted tags (<script>, <style>, <nav>, etc.) and extract plain text."""
        soup = BeautifulSoup(html_content, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.decompose()
        return soup.get_text(separator="\n", strip=True)

    def to_chunks(self, source: str, *, strategy: ChunkStrategy = ChunkStrategy.AUTO) -> list[str]:
        """Collect `source`, clean it and cut it into chunks. Makes no network call.

        `AUTO` picks `MARKDOWN` for a .md source and `SEMANTIC` for anything else.
        Returns an empty list when `source` is reachable but holds no text.

        Raises:
            SourceError: If `source` cannot be found, read, or is of an
                unsupported format.

        """
        text = self.clean(self._collect_file(source))
        if not text:
            return []
        if strategy is ChunkStrategy.AUTO:
            strategy = ChunkStrategy.MARKDOWN if source.lower().endswith(".md") else ChunkStrategy.SEMANTIC
        if strategy is ChunkStrategy.MARKDOWN:
            return self._titled_chunks(text)
        if strategy is ChunkStrategy.SENTENCE:
            return self.sentence_chunk(text)
        return self.semantic_chunk(text)

    def _titled_chunks(self, text: str) -> list[str]:
        """Cut every markdown section, prefixing its title to each chunk it yields.

        Without the title, "Its efficiency rarely exceeds 40%" carries no word
        tying it back to engines, for a keyword search or for an embedding.
        """
        chunks = []
        for section in self.markdown_chunk(text):
            chunks.extend(f"{section['title']}\n\n{chunk}" for chunk in self.semantic_chunk(section["text"]))
        return chunks

    def _normalize(self, data: str) -> str:
        return unicodedata.normalize("NFC", data)

    def _del_html(self, text: str) -> str:
        return _HTML_TAGS.sub("", text)

    def _del_inv_charac(self, text: str) -> str:
        return _INVISIBLE_CHARS.sub("", text)

    def _normalize_space(self, text: str) -> str:
        text = _SPACES.sub(" ", text)
        text = _BLANK_LINES.sub("\n\n", text)
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(lines).strip()

    def clean(self, text: str) -> str:
        """Normalise the text, drop HTML tags and invisible characters, tidy the spacing."""
        data = self._normalize(text)
        data = self._del_html(text=data)
        data = self._del_inv_charac(text=data)
        return self._normalize_space(text=data)

    def _get_sentences(self, text: str) -> list[str]:
        text = " ".join(text.split())
        if not text:
            return []
        return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]

    def sentence_chunk(self, text: str, *, overlap: int = _OVERLAP_SENTENCES) -> list[str]:
        """Cut the text into chunks of at most `_MAX_WORD_CHUNKS` words.

        The last `overlap` sentences of a chunk open the next one, so a fact
        sitting across a cut stays whole in at least one chunk. With `overlap`
        set to 0 the chunks are disjoint and every sentence appears once.
        """
        sentences = self._get_sentences(text)
        chunks: list[str] = []
        current: list[str] = []
        words = 0
        for sentence in sentences:
            length = len(sentence.split())
            if current and words + length > _MAX_WORD_CHUNKS:
                chunks.append(" ".join(current))
                current = current[len(current) - overlap :] if overlap else []
                words = sum(len(kept.split()) for kept in current)
            current.append(sentence)
            words += length
        if current:
            chunks.append(" ".join(current))
        return chunks

    def markdown_chunk(self, text: str) -> list[dict[str, str]]:
        """Create chunk based on the markdown structure, nor in the meaning of the doc."""
        sections: list[dict[str, str]] = []
        title, body = "(preamble)", []
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                if any(x.strip() for x in body):
                    sections.append({"title": title, "text": "\n".join(body).strip()})
                title, body = line.lstrip("#").strip(), []
            else:
                body.append(line)
        if any(x.strip() for x in body):
            sections.append({"title": title, "text": "\n".join(body).strip()})
        return sections

    def adaptive_threshold(self, scores: list[float]) -> float:
        """Breakpoint threshold: below the mean minus one standard deviation."""
        if len(scores) < _SENTENCE_MAX_NB:
            return 0.0
        return statistics.mean(scores) - statistics.pstdev(scores)

    def find_breakpoints(self, scores: list[float], threshold: float) -> list[int]:
        """Return the indices i such that a breakpoint occurs after sentence i."""
        return [i for i, s in enumerate(scores) if s < threshold]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Close in meaning between two vectors: 1.0 is identical, 0.0 unrelated, -1 is opposite meaning."""
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if not norm_a or not norm_b:
            return 0.0
        return sum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b)

    def consecutive_similarities(self, sentences: list[str]) -> list[float]:
        """Similarity between each sentence and the next one: len(sentences) - 1 scores."""
        if len(sentences) < _SENTENCE_MAX_NB:
            return []
        vectors = self.vectorize_batch(sentences)
        return [self._cosine_similarity(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]

    def split_on_breakpoints(self, sentences: list[str], breakpoints: set[int]) -> list[str]:
        """Group the sentences into chunks, cutting after each breakpoint index."""
        chunks, current = [], []
        for i, sentence in enumerate(sentences):
            current.append(sentence)
            if i in breakpoints:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
        return chunks

    def semantic_chunk(self, text: str) -> list[str]:
        """Split the text where its meaning shifts, rather than at a fixed word count."""
        sentences = self._get_sentences(text)
        if len(sentences) < _SENTENCE_MAX_NB:
            return sentences
        scores = self.consecutive_similarities(sentences)
        breakpoints = set(self.find_breakpoints(scores, self.adaptive_threshold(scores)))
        return self.split_on_breakpoints(sentences, breakpoints)

    @property
    def model(self) -> str:
        """Name of the embedding model currently in use."""
        return self._model

    def vectorize(self, text: str) -> list[float]:
        """Turn a text into a vector (embedding).

        Raises:
            EmbeddingError: If the provider returned no vector for the text.

        """
        return self.vectorize_batch([text])[0]

    def vectorize_batch(self, texts: list[str]) -> list[list[float]]:
        """Vectorize several texts, in as many requests as the provider's limit needs.

        A whole document in one request runs into the time the provider allows
        for it, and splitting it costs nothing measurable.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            vectors.extend(self._embed(texts[start : start + _EMBED_BATCH_SIZE]))
        return vectors

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed one request worth of texts, translating whatever the provider raises.

        Raises:
            EmbeddingError: If the provider failed to answer, or did not return
                one vector per text, which an unavailable embedding model is
                enough to cause. Provider errors are translated so that one
                failing source does not abort a whole batch of them.

        """
        try:
            vectors = self._embedder.embeddings(texts, self._model)
        except SensaiError as error:
            msg = f"model {self._model!r} could not embed {len(texts)} texts: {error}"
            raise EmbeddingError(msg) from error
        if not vectors:
            msg = f"model {self._model!r} produced no embedding: it may not be installed, or the provider is down"
            raise EmbeddingError(msg)
        if len(vectors) != len(texts):
            msg = f"model {self._model!r} returned {len(vectors)} vectors for {len(texts)} texts"
            raise EmbeddingError(msg)
        return vectors
