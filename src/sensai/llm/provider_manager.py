import contextlib
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
    _provider: ProviderAdapter
    _model: str | None = None

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

    def set_model(self, model: str) -> bool:
        """Set the model to use and unload the previous one. Returns True if successful, False otherwise."""
        provider = self._models.get(model)
        if provider is None:
            return False
        try:
            loaded = provider.load(model)
        except SensaiError:
            return False
        if not loaded:
            return False
        if self._model is not None and self._model != model:
            with contextlib.suppress(SensaiError):
                self._models[self._model].unload(self._model)
        self._provider = provider
        self._model = model
        return True

    def provider(self) -> ProviderAdapter:
        """Return the provider adapter in use."""
        return self._provider
