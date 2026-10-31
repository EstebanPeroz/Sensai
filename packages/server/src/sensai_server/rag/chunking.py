"""Clean a text and cut it into chunks, ready to be embedded and stored.

It stays sync: semantic chunking embeds every sentence, so async callers run it in a thread (`asyncio.to_thread`).
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sensai_server.llm.embedder import Embedder

_MAX_WORD_CHUNKS = 50
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


class ChunkStrategy(StrEnum):
    """How a text is cut into chunks."""

    AUTO = "auto"
    SENTENCE = "sentence"
    MARKDOWN = "markdown"
    SEMANTIC = "semantic"


class Chunker:
    """Turn a text into clean chunks, and chunks into vectors."""

    _embedder: Embedder

    def __init__(self, embedder: Embedder) -> None:
        """Keep the `embedder` turning texts into vectors."""
        self._embedder = embedder

    def to_chunks(self, text: str, name: str, *, strategy: ChunkStrategy = ChunkStrategy.AUTO) -> list[str]:
        """Clean `text` and cut it into chunks. `name` is the source it came from.

        `AUTO` picks `MARKDOWN` when `name` ends in .md and `SEMANTIC` otherwise.
        Returns an empty list when `text` holds nothing once cleaned.

        Raises:
            EmbeddingError: If the semantic strategy could not embed the sentences.

        """
        text = self.clean(text)
        if not text:
            return []
        if strategy is ChunkStrategy.AUTO:
            strategy = ChunkStrategy.MARKDOWN if name.lower().endswith(".md") else ChunkStrategy.SEMANTIC
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
        """Cut the text on its markdown headings, not on the meaning of the document."""
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

    def vectorize_batch(self, texts: list[str]) -> list[list[float]]:
        """Vectorize several texts, in as many requests as the provider's limit needs.

        A whole document in one request runs into the time the provider allows
        for it, and splitting it costs nothing measurable.

        Raises:
            EmbeddingError: If the embedder failed, or did not return one vector per text.

        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            vectors.extend(self._embedder.embed(texts[start : start + _EMBED_BATCH_SIZE]))
        return vectors
