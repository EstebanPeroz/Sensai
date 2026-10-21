from typing import ClassVar, override

import pytest

from sensai.config.settings import LLMSettings, OllamaSettings
from sensai.error import SensaiError
from sensai.llm import provider_manager
from sensai.llm.adapter import ProviderAdapter
from sensai.llm.provider_manager import ProviderManager


class FakeAdapter(ProviderAdapter):
    models: ClassVar[list[str]] = ["a", "b"]
    refused: ClassVar[set[str]] = set()
    failing: ClassVar[set[str]] = set()

    def __init__(self, _settings: OllamaSettings) -> None:
        self.calls: list[tuple[str, str]] = []

    @override
    def chat(self, payload: dict, *, stream: bool) -> None:
        raise NotImplementedError

    @override
    def embedding(self, message: str, model: str) -> list:
        raise NotImplementedError

    @override
    def embeddings(self, messages: list[str], model: str) -> list[list]:
        raise NotImplementedError

    @override
    def show(self, model_name: str) -> None:
        raise NotImplementedError

    @override
    def load(self, model_name: str) -> bool:
        self.calls.append(("load", model_name))
        if model_name in self.failing:
            raise SensaiError(model_name)
        if model_name in self.refused:
            return False
        self._current_model = model_name
        return True

    @override
    def unload(self, model_name: str) -> bool:
        self.calls.append(("unload", model_name))
        if model_name in self.failing:
            raise SensaiError(model_name)
        if self._current_model == model_name:
            self._current_model = None
        return True

    @override
    def list(self) -> list[str]:
        return self.models


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> ProviderManager:
    monkeypatch.setitem(provider_manager.PROVIDER_ADAPTERS, "ollama", FakeAdapter)
    monkeypatch.setattr(FakeAdapter, "refused", set())
    monkeypatch.setattr(FakeAdapter, "failing", set())
    return ProviderManager(LLMSettings(providers={"ollama": OllamaSettings()}))


def calls(manager: ProviderManager) -> list[tuple[str, str]]:
    adapter = manager.provider()
    assert isinstance(adapter, FakeAdapter)
    return adapter.calls


class TestResolve:
    def test_a_name_a_provider_lists_is_its_own(self, manager: ProviderManager) -> None:
        assert manager.resolve("a") == "a"

    def test_an_untagged_name_finds_the_latest_tag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(FakeAdapter, "models", ["nomic-embed-text:latest"])
        monkeypatch.setitem(provider_manager.PROVIDER_ADAPTERS, "ollama", FakeAdapter)
        resolving = ProviderManager(LLMSettings(providers={"ollama": OllamaSettings()}))

        assert resolving.resolve("nomic-embed-text") == "nomic-embed-text:latest"

    def test_a_tagged_name_is_not_tagged_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(FakeAdapter, "models", ["qwen2.5:1.5b"])
        monkeypatch.setitem(provider_manager.PROVIDER_ADAPTERS, "ollama", FakeAdapter)
        resolving = ProviderManager(LLMSettings(providers={"ollama": OllamaSettings()}))

        assert resolving.resolve("qwen2.5:1.5b") == "qwen2.5:1.5b"
        assert resolving.resolve("qwen2.5") is None

    def test_a_name_no_provider_serves_resolves_to_nothing(self, manager: ProviderManager) -> None:
        assert manager.resolve("missing") is None


class TestGetProviderResolves:
    def test_an_untagged_name_reaches_the_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(FakeAdapter, "models", ["nomic-embed-text:latest"])
        monkeypatch.setitem(provider_manager.PROVIDER_ADAPTERS, "ollama", FakeAdapter)
        resolving = ProviderManager(LLMSettings(providers={"ollama": OllamaSettings()}))

        assert resolving.get_provider("nomic-embed-text") is not None


class TestSetModelResolves:
    @pytest.fixture
    def tagged(self, monkeypatch: pytest.MonkeyPatch) -> ProviderManager:
        """A manager over a provider listing one model under its `:latest` tag."""
        monkeypatch.setattr(FakeAdapter, "models", ["nomic-embed-text:latest"])
        monkeypatch.setattr(FakeAdapter, "refused", set())
        monkeypatch.setattr(FakeAdapter, "failing", set())
        monkeypatch.setitem(provider_manager.PROVIDER_ADAPTERS, "ollama", FakeAdapter)
        return ProviderManager(LLMSettings(providers={"ollama": OllamaSettings()}))

    def test_an_untagged_name_is_accepted(self, tagged: ProviderManager) -> None:
        assert tagged.set_model("nomic-embed-text")

    def test_the_provider_is_asked_for_the_tagged_name(self, tagged: ProviderManager) -> None:
        tagged.set_model("nomic-embed-text")

        assert calls(tagged) == [("load", "nomic-embed-text:latest")]

    def test_the_same_model_under_both_names_is_not_reloaded(self, tagged: ProviderManager) -> None:
        tagged.set_model("nomic-embed-text")

        assert tagged.set_model("nomic-embed-text:latest")
        assert ("unload", "nomic-embed-text:latest") not in calls(tagged)


class TestSetModel:
    def test_first_model_unloads_nothing(self, manager: ProviderManager) -> None:
        assert manager.set_model("a")

        assert calls(manager) == [("load", "a")]
        assert manager.provider().current_model() == "a"

    def test_switch_unloads_previous_and_keeps_new_active(self, manager: ProviderManager) -> None:
        manager.set_model("a")

        assert manager.set_model("b")

        assert calls(manager) == [("load", "a"), ("load", "b"), ("unload", "a")]
        assert manager.provider().current_model() == "b"

    def test_selecting_current_model_does_not_unload_it(self, manager: ProviderManager) -> None:
        manager.set_model("a")

        assert manager.set_model("a")

        assert calls(manager) == [("load", "a"), ("load", "a")]
        assert manager.provider().current_model() == "a"

    def test_unknown_model_keeps_previous(self, manager: ProviderManager) -> None:
        manager.set_model("a")

        assert not manager.set_model("missing")

        assert calls(manager) == [("load", "a")]
        assert manager.provider().current_model() == "a"

    def test_refused_load_keeps_previous(self, manager: ProviderManager) -> None:
        FakeAdapter.refused = {"b"}
        manager.set_model("a")

        assert not manager.set_model("b")

        assert calls(manager) == [("load", "a"), ("load", "b")]
        assert manager.provider().current_model() == "a"

    def test_failing_load_keeps_previous(self, manager: ProviderManager) -> None:
        FakeAdapter.failing = {"b"}
        manager.set_model("a")

        assert not manager.set_model("b")

        assert calls(manager) == [("load", "a"), ("load", "b")]
        assert manager.provider().current_model() == "a"

    def test_failing_unload_does_not_cancel_switch(self, manager: ProviderManager) -> None:
        manager.set_model("a")
        FakeAdapter.failing = {"a"}

        assert manager.set_model("b")

        assert calls(manager) == [("load", "a"), ("load", "b"), ("unload", "a")]
        assert manager.provider().current_model() == "b"
