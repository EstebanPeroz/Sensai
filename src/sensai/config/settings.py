from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OllamaSettings:
    """Connection settings of the Ollama HTTP API."""

    base_url: str = "http://localhost:11434/"
    embedding_model: str = "nomic-embed-text"
    timeout: float = 30.0
