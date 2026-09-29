from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from sensai.config.settings import PROVIDERS, OllamaSettings, ProviderSettings
from sensai.llm.adapter import ProviderAdapter
from sensai.llm.ollama import OllamaAdapter
from sensai.llm.registry import ModelError, ModelRef, ProviderRegistry


def fake_adapter(models: list[str], default: str | None = None) -> MagicMock:
    adapter = MagicMock(spec=ProviderAdapter)
    adapter.available_models.return_value = models
    adapter.get_default_model.return_value = default
    return adapter


@pytest.fixture
def registry() -> ProviderRegistry:
    return ProviderRegistry(
        {
            "ollama": fake_adapter(["qwen2.5:1.5b", "llama3.2"], default="qwen2.5:1.5b"),
            "mistral": fake_adapter(["mistral-small"], default="mistral-small"),
        },
    )


class TestFromSettings:
    def test_builds_an_adapter_per_provider(self) -> None:
        registry = ProviderRegistry.from_settings([OllamaSettings()])

        assert isinstance(registry.adapter("ollama"), OllamaAdapter)

    def test_every_supported_provider_has_an_adapter(self) -> None:
        registry = ProviderRegistry.from_settings([provider() for provider in PROVIDERS])

        assert all(isinstance(registry.adapter(provider.name), ProviderAdapter) for provider in PROVIDERS)

    def test_rejects_provider_without_adapter(self) -> None:
        @dataclass
        class UnknownSettings(ProviderSettings):
            name: ClassVar[str] = "unknown"

        with pytest.raises(TypeError, match="no adapter for provider 'unknown'"):
            ProviderRegistry.from_settings([UnknownSettings()])


class TestAvailableModels:
    def test_concatenates_models_of_every_provider(self, registry: ProviderRegistry) -> None:
        assert registry.available_models() == [
            ModelRef("ollama", "qwen2.5:1.5b"),
            ModelRef("ollama", "llama3.2"),
            ModelRef("mistral", "mistral-small"),
        ]

    def test_is_empty_without_provider(self) -> None:
        assert ProviderRegistry({}).available_models() == []


class TestSelectModel:
    def test_selects_model_served_by_a_provider(self, registry: ProviderRegistry) -> None:
        assert registry.select_model("mistral-small") == ModelRef("mistral", "mistral-small")

    def test_rejects_unknown_model(self, registry: ProviderRegistry) -> None:
        with pytest.raises(ModelError) as err:
            registry.select_model("qwen3")

        assert str(err.value) == "unknown model 'qwen3', available models: qwen2.5:1.5b, llama3.2, mistral-small"

    def test_defaults_to_first_provider_default_model(self, registry: ProviderRegistry) -> None:
        assert registry.select_model(None) == ModelRef("ollama", "qwen2.5:1.5b")

    def test_skips_providers_without_default_model(self) -> None:
        registry = ProviderRegistry({"ollama": fake_adapter([]), "mistral": fake_adapter(["m"], default="m")})

        assert registry.select_model(None) == ModelRef("mistral", "m")

    def test_fails_when_no_provider_has_a_model(self) -> None:
        with pytest.raises(ModelError, match="no model available"):
            ProviderRegistry({"ollama": fake_adapter([])}).select_model(None)
