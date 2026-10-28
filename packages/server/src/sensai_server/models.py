from __future__ import annotations

from typing import Protocol

DEFAULT_MODELS = ["qwen3:1.7b"]  # replace with the providerManager list once the rest of the src impl


class ModelCatalog(Protocol):
    """Tells which models a client can log in with."""

    async def available_models(self) -> list[str]:
        """Return the names of the models a provider serves."""
        ...


# TEMPORARY: replaced by a catalog backed by `ProviderManager` once the providers are copied from `src/`.
class StaticModelCatalog:
    """A fixed list of models, used until the providers are copied from `src/`."""

    _models: list[str]

    def __init__(self) -> None:
        """Serve exactly `models`."""
        self._models = list(DEFAULT_MODELS)

    async def available_models(self) -> list[str]:
        """Return the names of the models a provider serves."""
        return list(self._models)
