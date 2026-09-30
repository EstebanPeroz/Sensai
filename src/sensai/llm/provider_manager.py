from typing import TYPE_CHECKING

from sensai.llm.ollama import OllamaAdapter

if TYPE_CHECKING:
    from sensai.llm.adapter import ProviderAdapter


class ProviderManager:
    """Aggregates the registered LLM providers and maps each model name to the provider that serves it."""

    _list: list[ProviderAdapter]
    _models: dict[str, ProviderAdapter]

    def __init__(self) -> None:
        """Register the known providers and index every model they each expose."""
        self._list = []
        self._models = {}

        self._list.append(OllamaAdapter())

        for provider in self._list:
            models = provider.list()
            for model in models:
                self._models[model] = provider

    def get_models(self) -> list[str]:
        """List the names of every model available across all registered providers."""
        return list(self._models.keys())

    def get_provider(self, model: str) -> ProviderAdapter | None:
        """Return the provider serving `model`, or None if no registered provider exposes it."""
        return self._models.get(model)

    def get_base_model(self) -> tuple[str, ProviderAdapter] | None:
        """Return the default model name and its provider, or None if it isn't available."""
        model = "qwen3:1.7b"
        provider = self._models.get(model)
        if provider is not None:
            return model, provider
        return None

    def set_model(self, model: str) -> bool:
        """Set the model to use. Returns True if successful, False if the model is not available."""
        provider = self._models.get(model)
        if provider is not None:
            provider.load(model)
            return True
        return False
