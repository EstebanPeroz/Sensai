from typing import TYPE_CHECKING

from sensai.llm.ollama import OllamaAdapter

if TYPE_CHECKING:
    from sensai.llm.adapter import ProviderAdapter


class ProviderManager:
    """TMP."""

    _list: list[ProviderAdapter]
    _models: dict[str, ProviderAdapter]

    def __init__(self) -> None:
        """TMP."""
        self._list = []
        self._models = {}

        self._list.append(OllamaAdapter())

        for provider in self._list:
            models = provider.list()
            for model in models:
                self._models[model] = provider

    def get_models(self) -> list[str]:
        """TMP."""
        return list(self._models.keys())

    def get_provider(self, model: str) -> ProviderAdapter | None:
        """TMP."""
        return self._models.get(model)
