"""What the benchmarks read straight from Ollama, outside of Sensai: the size of a model and where it is loaded."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from sensai.config.settings import Settings

TIMEOUT = 5


def size_gb(settings: Settings, model: str) -> float | None:
    """Size of `model` on disk, as Ollama lists it, None when it cannot tell."""
    sizes = [entry["size"] for entry in models(settings, "api/tags") if entry.get("name") == model]
    return round(sizes[0] / 1e9, 2) if sizes else None


def gpu_share(settings: Settings, model: str) -> float | None:
    """Share of the loaded `model` that sits in the memory of the GPU, None when it is not loaded."""
    loaded = [entry for entry in models(settings, "api/ps") if entry.get("name") == model]
    if not loaded or not loaded[0].get("size"):
        return None
    return round(loaded[0].get("size_vram", 0) / loaded[0]["size"], 2)


def models(settings: Settings, endpoint: str) -> list[dict[str, Any]]:
    """Models listed by the Ollama `endpoint`, `api/tags` for the installed ones or `api/ps` for the loaded ones."""
    provider = settings.llm.providers.get("ollama")
    if provider is None:
        return []
    try:
        response = requests.get(f"{provider.base_url.rstrip('/')}/{endpoint}", timeout=TIMEOUT)
        return response.json().get("models", [])
    except requests.RequestException, ValueError:
        return []
