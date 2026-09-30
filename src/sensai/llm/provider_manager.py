from typing import TYPE_CHECKING

from sensai.error import SensaiError
from sensai.llm.ollama import OllamaAdapter

if TYPE_CHECKING:
    from sensai.config.settings import LLMSettings
    from sensai.llm.adapter import ProviderAdapter


PROVIDER_ADAPTERS: dict[str, type] = {
    "ollama": OllamaAdapter,
}


class ProviderManager:
    """Aggregates the registered LLM providers and maps each model name to the provider that serves it."""

    _list: list[ProviderAdapter]
    _models: dict[str, ProviderAdapter]

    def __init__(self, settings: LLMSettings) -> None:
        """Register the known providers and index every model they each expose."""
        self._list = []
        self._models = {}

        for name, adapter_cls in PROVIDER_ADAPTERS.items():
            provider_conf = settings.providers.get(name)
            if provider_conf is None:
                continue
            provider: ProviderAdapter = adapter_cls(provider_conf)
            try:
                self._list.append(provider)
                for model in provider.list():
                    self._models[model] = provider
            except SensaiError:
                # Add log to fail ollama connection
                continue

        if not self._list:
            msg = f"no configured provider found, expected one of: {list(PROVIDER_ADAPTERS)}"
            raise SensaiError(msg)

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
