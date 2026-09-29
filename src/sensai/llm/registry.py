from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from sensai.config.settings import OllamaSettings
from sensai.llm.ollama import OllamaAdapter

if TYPE_CHECKING:
    from sensai.config.settings import ProviderSettings
    from sensai.llm.adapter import ProviderAdapter


class ModelError(Exception):
    """Raised when the requested model, or any model, cannot be selected."""


class ModelRef(NamedTuple):
    """A model and the provider serving it."""

    provider: str
    name: str


class ProviderRegistry:
    """Adapters of every configured provider, looked up by provider name."""

    def __init__(self, adapters: dict[str, ProviderAdapter]) -> None:
        """Init the registry from adapters keyed by provider name, in priority order."""
        self._adapters: dict[str, ProviderAdapter] = adapters

    @classmethod
    def from_settings(cls, providers: list[ProviderSettings]) -> ProviderRegistry:
        """Build the adapter of every configured provider."""
        return cls({provider.name: _build_adapter(provider) for provider in providers})

    def adapter(self, provider: str) -> ProviderAdapter:
        """Return the adapter of a provider."""
        return self._adapters[provider]

    def available_models(self) -> list[ModelRef]:
        """Fetch the models of every provider from its API, in provider order."""
        return [
            ModelRef(provider, model)
            for provider, adapter in self._adapters.items()
            for model in adapter.available_models()
        ]

    def select_model(self, model: str | None) -> ModelRef:
        """Return model if a provider serves it, or the default model of the first provider having one if None."""
        if model is None:
            return self._default_model()
        available = self.available_models()
        for ref in available:
            if ref.name == model:
                return ref
        names = ", ".join(ref.name for ref in available) or "none"
        msg = f"unknown model {model!r}, available models: {names}"
        raise ModelError(msg)

    def _default_model(self) -> ModelRef:
        for provider, adapter in self._adapters.items():
            model = adapter.get_default_model()
            if model is not None:
                return ModelRef(provider, model)
        msg = "no model available from the configured providers"
        raise ModelError(msg)


def _build_adapter(settings: ProviderSettings) -> ProviderAdapter:
    match settings:
        case OllamaSettings():
            return OllamaAdapter(settings)
    msg = f"no adapter for provider {settings.name!r}"
    raise TypeError(msg)
