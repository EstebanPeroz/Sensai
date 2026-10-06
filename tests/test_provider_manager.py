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
