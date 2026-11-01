import pytest
from sensai_protocol.error import SensaiError
from sensai_server.config.settings import LLMSettings, OllamaSettings
from sensai_server.llm.provider import OllamaAdapter, ProviderManager, provider_manager

from tests.server.fakes import FakeAdapter

SETTINGS = LLMSettings(providers={"ollama": OllamaSettings(base_url="http://ollama:1234")})


def fake(manager: ProviderManager, model: str) -> FakeAdapter:
    adapter = manager.get_provider(model)
    assert isinstance(adapter, FakeAdapter)
    return adapter


class TestRegistry:
    def test_ollama_is_the_registered_adapter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.undo()

        assert {"ollama": OllamaAdapter} == provider_manager.PROVIDER_ADAPTERS

    def test_every_registered_adapter_has_settings(self) -> None:
        from sensai_server.config.settings import PROVIDERS  # noqa: PLC0415

        assert set(provider_manager.PROVIDER_ADAPTERS) <= {provider.name for provider in PROVIDERS}


class TestInit:
    def test_adapter_gets_its_provider_settings(self) -> None:
        manager = ProviderManager(SETTINGS)

        assert fake(manager, "a").settings == OllamaSettings(base_url="http://ollama:1234")

    def test_no_configured_provider_raises(self) -> None:
        with pytest.raises(SensaiError, match="no configured provider found"):
            ProviderManager(LLMSettings())

    def test_unreachable_provider_serves_no_model(self) -> None:
        FakeAdapter.unreachable = True

        manager = ProviderManager(SETTINGS)

        assert manager.get_models() == []
        assert manager.get_provider("a") is None


class TestModels:
    def test_lists_every_model(self) -> None:
        assert ProviderManager(SETTINGS).get_models() == ["a", "b"]

    def test_models_share_their_provider(self) -> None:
        manager = ProviderManager(SETTINGS)

        assert manager.get_provider("a") is manager.get_provider("b")

    def test_unknown_model_has_no_provider(self) -> None:
        assert ProviderManager(SETTINGS).get_provider("missing") is None


class TestGetProvider:
    def test_untagged_model_resolves_to_latest(self) -> None:
        FakeAdapter.models = ["qwen3:1.7b", "nomic-embed-text:latest"]
        manager = ProviderManager(SETTINGS)

        assert manager.get_provider("nomic-embed-text") is not None
        assert manager.get_provider("qwen3:1.7b") is not None
        assert manager.get_provider("qwen3") is None


class TestLoadModel:
    def test_loads_on_the_serving_provider(self) -> None:
        manager = ProviderManager(SETTINGS)

        assert manager.load_model("b")

        assert fake(manager, "b").calls == [("load", "b")]

    def test_unknown_model_loads_nothing(self) -> None:
        manager = ProviderManager(SETTINGS)

        assert not manager.load_model("missing")

        assert fake(manager, "a").calls == []

    def test_refused_load(self) -> None:
        FakeAdapter.refused = {"b"}
        manager = ProviderManager(SETTINGS)

        assert not manager.load_model("b")

    def test_failing_load_is_caught(self) -> None:
        FakeAdapter.failing = {"b"}
        manager = ProviderManager(SETTINGS)

        assert not manager.load_model("b")
        assert fake(manager, "b").calls == [("load", "b")]
