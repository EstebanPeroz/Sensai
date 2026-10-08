from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sensai.llm.embedder import Embedder

# Rough average for English text with the usual BPE tokenizers.
CHARS_PER_TOKEN = 4
TRUNCATION_MARK = " […]"

_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BARE_URL = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    """Drop what costs tokens without informing the model: images, link targets, bare URLs, extra whitespace."""
    text = _MARKDOWN_IMAGE.sub("", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _BARE_URL.sub("", text)
    return " ".join(text.split())


def chunk_words(text: str, size: int) -> list[str]:
    """Split `text` into consecutive chunks of about `size` words each."""
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors, 0 when either one is null."""
    norm = math.hypot(*a) * math.hypot(*b)
    if norm == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm


def top_chunks(chunks: list[str], query: str, embedder: Embedder, count: int) -> list[str]:
    """Keep the `count` chunks closest to `query`, in their original order.

    Without embeddings (no embedding model available), the first `count` chunks are kept.
    """
    if len(chunks) <= count:
        return chunks
    vectors = embedder.embed([query, *chunks])
    if not vectors:
        return chunks[:count]
    query_vector, chunk_vectors = vectors[0], vectors[1:]
    scores = [cosine(query_vector, vector) for vector in chunk_vectors]
    best = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:count]
    return [chunks[i] for i in sorted(best)]


def truncate_tokens(text: str, max_tokens: int) -> str:
    """Cut `text` so it holds about `max_tokens` tokens, marking the cut."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(TRUNCATION_MARK)].rstrip() + TRUNCATION_MARK
